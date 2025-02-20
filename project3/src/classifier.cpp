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

// Structure to hold a training sample with its label and feature vector.
struct TrainingSample
{
    std::string label;
    std::vector<double> features;
};

// Function to load training data from a CSV file.
// Expected CSV format per line: label,feature1,feature2
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
        // Read the full row into a temporary vector
        std::vector<double> allFeatures;
        while (std::getline(ss, token, ','))
        {
            allFeatures.push_back(std::stod(token));
        }
        // Adjust to keep only percentFilled and heightWidthRatio
        if (allFeatures.size() >= 3)
        {
            // Remove bestArea (at index 0) and keep indices 1 and 2.
            sample.features.push_back(allFeatures[1]); // percentFilled
            sample.features.push_back(allFeatures[2]); // heightWidthRatio
        }
        samples.push_back(sample);
    }
    return samples;
}

// Compute features from the input image.
// The method mimics the processing in ImageProcessor by first preprocessing,
// dynamically selecting a threshold, thresholding and cleaning up the binary image,
// and then performing connected components analysis to extract features (percentFilled and heightWidthRatio)
// for the largest object found.
std::vector<double> computeFeatures(const cv::Mat &image)
{
    ImageProcessor ip;

    // Preprocess the image (convert to grayscale, blur, and adjust based on saturation, etc.)
    cv::Mat processed = ip.preprocess_image(image);

    // Compute a dynamic threshold using a k-means based method
    int threshold_value = ip.get_dynamic_threshold(processed);

    // Apply custom thresholding
    cv::Mat binary = Thresholder::apply(processed, threshold_value);

    // Clean up the binary image using morphological filtering
    binary = Morphology::closing(binary, 7);
    binary = Morphology::closing(binary, 7);
    binary = Morphology::opening(binary, 3);

    // Compute connected components to extract regions
    cv::Mat labels, stats, centroids;
    int numLabels = cv::connectedComponentsWithStats(binary, labels, stats, centroids, 8, CV_32S);

    // Find the largest region (ignoring background, label 0)
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
        std::cerr << "No valid object region found in the image." << std::endl;
        return {};
    }

    // Get bounding box info for the selected region
    int left = stats.at<int>(bestLabel, cv::CC_STAT_LEFT);
    int top = stats.at<int>(bestLabel, cv::CC_STAT_TOP);
    int width = stats.at<int>(bestLabel, cv::CC_STAT_WIDTH);
    int height = stats.at<int>(bestLabel, cv::CC_STAT_HEIGHT);
    double boundingBoxArea = static_cast<double>(width * height);

    // Compute features:
    // 1. Percent filled = region area divided by the area of its bounding box
    double percentFilled = static_cast<double>(bestArea) / boundingBoxArea;
    // 2. Height/Width ratio
    double heightWidthRatio = static_cast<double>(height) / width;

    std::vector<double> features = {percentFilled, heightWidthRatio};
    return features;
}

// Compute the scaled Euclidean distance between two feature vectors.
// Each difference is normalized by the standard deviation for that feature.
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
        std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return -1;
    }

    std::string imagePath = argv[1];
    cv::Mat image = cv::imread(imagePath);
    if (image.empty())
    {
        std::cerr << "Could not load image: " << imagePath << std::endl;
        return -1;
    }

    // Compute feature vector for the input image
    std::vector<double> testFeatures = computeFeatures(image);
    if (testFeatures.empty())
    {
        std::cerr << "Failed to compute features from the image." << std::endl;
        return -1;
    }

    // Load training data from CSV file (assumed to be named "training_data.csv" in the working directory)
    std::vector<TrainingSample> trainingSamples = loadTrainingData("training_data.csv");
    if (trainingSamples.empty())
    {
        std::cerr << "No training data found." << std::endl;
        return -1;
    }

    size_t featureDim = testFeatures.size();
    // Compute the mean and standard deviation for each feature across the training data.
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

    // Classify the test image using a nearest-neighbor approach based on scaled Euclidean distance.
    double minDistance = std::numeric_limits<double>::max();
    std::string predictedLabel = "Unknown";
    for (const auto &sample : trainingSamples)
    {
        double distance = computeScaledDistance(testFeatures, sample.features, stdDevs);
        if (distance < minDistance)
        {
            minDistance = distance;
            predictedLabel = sample.label;
        }
    }

    // Display the image with the predicted label at the top left corner with increased font size.
    cv::putText(image, predictedLabel, cv::Point(15, 75),
                cv::FONT_HERSHEY_SIMPLEX, 3.0, cv::Scalar(0, 0, 255), 4);
    cv::imshow("Classified Image", image);
    cv::waitKey(0);

    return 0;
}