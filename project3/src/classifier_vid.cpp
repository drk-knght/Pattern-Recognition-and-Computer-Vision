#include "ImageProcessor.h"
#include "Thresholder.h"
#include "morphology.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <limits>
#include <algorithm>

// Structure to hold a training sample with its label and feature vector.
struct TrainingSample
{
    std::string label;
    std::vector<double> features;
};

// Function to load training data from a CSV file.
// Expected CSV format per line:
//    label,bestArea,percentFilled,heightWidthRatio,centroid_x,centroid_y,left,top,width,height
// This function uses only two features (percentFilled and heightWidthRatio).
std::vector<TrainingSample> loadTrainingData(const std::string &filename)
{
    std::vector<TrainingSample> samples;
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Could not open training data file: " << filename << std::endl;
        return samples;
    }
    std::string line;
    while (std::getline(file, line))
    {
        if (line.empty())
            continue;
        std::istringstream ss(line);
        std::string token;
        TrainingSample sample;
        // Read label first
        if (std::getline(ss, token, ','))
        {
            sample.label = token;
        }
        // Read remaining features
        std::vector<double> allFeatures;
        while (std::getline(ss, token, ','))
        {
            allFeatures.push_back(std::stod(token));
        }
        // Keep only percentFilled and heightWidthRatio (assumed to be at indices 1 and 2)
        if (allFeatures.size() >= 3)
        {
            sample.features.push_back(allFeatures[1]); // percentFilled
            sample.features.push_back(allFeatures[2]); // heightWidthRatio
        }
        samples.push_back(sample);
    }
    return samples;
}

// Compute features from the input frame.
// This mimics the processing in ImageProcessor and classifier.cpp.
std::vector<double> computeFeatures(const cv::Mat &image)
{
    ImageProcessor ip;
    cv::Mat processed = ip.preprocess_image(image);
    int threshold_value = ip.get_dynamic_threshold(processed);
    cv::Mat binary = Thresholder::apply(processed, threshold_value);
    // Apply morphological filtering as in your classifier
    binary = Morphology::closing(binary, 7);
    binary = Morphology::closing(binary, 7);
    binary = Morphology::opening(binary, 3);

    // Use connected components to extract features from the largest object.
    cv::Mat labels, stats, centroids;
    int numLabels = cv::connectedComponentsWithStats(binary, labels, stats, centroids, 8, CV_32S);

    int bestLabel = -1;
    int bestArea = 0;
    for (int i = 1; i < numLabels; i++)
    {
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area > bestArea)
        {
            bestArea = area;
            bestLabel = i;
        }
    }

    if (bestLabel == -1)
    {
        std::cerr << "No valid object region found in frame." << std::endl;
        return {};
    }

    int left = stats.at<int>(bestLabel, cv::CC_STAT_LEFT);
    int top = stats.at<int>(bestLabel, cv::CC_STAT_TOP);
    int width = stats.at<int>(bestLabel, cv::CC_STAT_WIDTH);
    int height = stats.at<int>(bestLabel, cv::CC_STAT_HEIGHT);
    double boundingBoxArea = static_cast<double>(width * height);
    double percentFilled = static_cast<double>(bestArea) / boundingBoxArea;
    double heightWidthRatio = static_cast<double>(height) / width;

    return {percentFilled, heightWidthRatio};
}

// Compute the scaled Euclidean distance between two feature vectors.
double computeScaledDistance(const std::vector<double> &features1,
                             const std::vector<double> &features2,
                             const std::vector<double> &stdDevs)
{
    double sum = 0.0;
    for (size_t i = 0; i < features1.size(); i++)
    {
        double diff = (features1[i] - features2[i]) / (stdDevs[i] == 0 ? 1 : stdDevs[i]);
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <video_path>" << std::endl;
        return -1;
    }

    std::string videoPath = argv[1];
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened())
    {
        std::cerr << "Error opening video: " << videoPath << std::endl;
        return -1;
    }

    // Load training data from the CSV file.
    std::vector<TrainingSample> trainingSamples = loadTrainingData("training_data.csv");
    if (trainingSamples.empty())
    {
        std::cerr << "No training data found." << std::endl;
        return -1;
    }

    // Compute standard deviations for each feature based on the training data.
    size_t featureDim = trainingSamples[0].features.size();
    std::vector<double> means(featureDim, 0.0), stdDevs(featureDim, 0.0);
    for (const auto &sample : trainingSamples)
    {
        for (size_t i = 0; i < featureDim; i++)
        {
            means[i] += sample.features[i];
        }
    }
    for (size_t i = 0; i < featureDim; i++)
    {
        means[i] /= trainingSamples.size();
    }
    for (const auto &sample : trainingSamples)
    {
        for (size_t i = 0; i < featureDim; i++)
        {
            double diff = sample.features[i] - means[i];
            stdDevs[i] += diff * diff;
        }
    }
    for (size_t i = 0; i < featureDim; i++)
    {
        stdDevs[i] = std::sqrt(stdDevs[i] / trainingSamples.size());
        if (stdDevs[i] == 0)
            stdDevs[i] = 1; // Avoid division by zero.
    }

    // Retrieve video properties for logging.
    double totalFrames = cap.get(cv::CAP_PROP_FRAME_COUNT);
    double fps = cap.get(cv::CAP_PROP_FPS);
    double totalDuration = totalFrames / fps;
    std::cout << "Video details: " << totalFrames << " frames, FPS: " << fps
              << ", Duration: " << totalDuration << " seconds." << std::endl;

    // Create video window with fixed size 600x400.
    cv::namedWindow("Video Classification", cv::WINDOW_NORMAL);
    cv::resizeWindow("Video Classification", 600, 400);

    // Process the video stream frame by frame.
    while (true)
    {
        cv::Mat frame;
        bool ret = cap.read(frame);
        if (!ret)
        {
            break; // End of video.
        }

        // Log details about the video progress.
        double currentFrame = cap.get(cv::CAP_PROP_POS_FRAMES);
        double currentTime = currentFrame / fps;
        double remainingFrames = totalFrames - currentFrame;
        double remainingTime = totalDuration - currentTime;
        std::cout << "Progress: Played " << currentFrame << " / "
                  << totalFrames << " frames (" << currentTime << "s / "
                  << totalDuration << "s); Remaining: " << remainingFrames
                  << " frames (" << remainingTime << "s)." << std::endl;

        // Compute feature vector for the current frame.
        std::vector<double> testFeatures = computeFeatures(frame);
        std::string predictedLabel = "Unknown";
        double minDistance = std::numeric_limits<double>::max();

        if (!testFeatures.empty())
        {
            // Classify using a nearest-neighbor approach.
            for (const auto &sample : trainingSamples)
            {
                double distance = computeScaledDistance(testFeatures, sample.features, stdDevs);
                if (distance < minDistance)
                {
                    minDistance = distance;
                    predictedLabel = sample.label;
                }
            }
        }

        // Overlay the predicted label on the video frame at the top left.
        int fontFace = cv::FONT_HERSHEY_SIMPLEX;
        double fontScale = 1.0;
        int thickness = 2;
        int baseline = 0;
        cv::Size textSize = cv::getTextSize(predictedLabel, fontFace, fontScale, thickness, &baseline);
        cv::Point textOrg(10, textSize.height + 10); // Position at top left.
        cv::putText(frame, predictedLabel, textOrg, fontFace, fontScale, cv::Scalar(255, 0, 0), thickness);

        cv::imshow("Video Classification", frame);

        // Reduced waitKey delay to speed up frame processing.
        char c = (char)cv::waitKey(1);
        if (c == 27 || c == 'q' || c == 'Q')
        { // Exit if ESC or q is pressed.
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
