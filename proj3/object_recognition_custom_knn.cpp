/*
    ADD HEADER
*/

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <map>
#include <fstream>
#include <limits>
#include <cmath>
#include <iostream>
#include <filesystem>
#include <algorithm>
#include <string>
#include <sstream>
#include <vector>
#include <unordered_map>

// Comparator for the map
struct Point2dComparator
{
    bool operator()(const cv::Point2d &p1, const cv::Point2d &p2) const
    {
        if (p1.x < p2.x)
            return true;
        if (p1.x > p2.x)
            return false;
        return p1.y < p2.y;
    }
};

std::map<std::string, std::vector<double>> trainingData;

// Function to compute the scaled Euclidean distance between two feature vectors
std::vector<double> computeFeatures(const cv::Mat &labels, const cv::Mat &stats, const cv::Mat &centroids, int label, cv::Mat &dst)
{
    // Compute the axis of least central moment
    cv::Moments moments = cv::moments(labels == label);
    double huMoments[7];
    cv::HuMoments(moments, huMoments); // Hu Moments are invariant to scale, rotation, and translation

    // Compute the orientation of the object
    double angle = 0.5 * atan2(2 * moments.mu11, moments.mu20 - moments.mu02);

    // Compute the oriented bounding box
    std::vector<cv::Point> points;
    for (int y = 0; y < labels.rows; ++y)
    {
        for (int x = 0; x < labels.cols; ++x)
        {
            if (labels.at<int>(y, x) == label)
            {
                points.push_back(cv::Point(x, y));
            }
        }
    }
    cv::RotatedRect rotatedRect = cv::minAreaRect(points);

    // Draw the oriented bounding box
    cv::Point2f vertices[4];
    rotatedRect.points(vertices);
    for (int i = 0; i < 4; i++)
        cv::line(dst, vertices[i], vertices[(i + 1) % 4], cv::Scalar(0, 255, 0));

    // Draw the orientation line
    cv::Point2d centroid(centroids.at<double>(label, 0), centroids.at<double>(label, 1));
    cv::Point2d delta(100 * cos(angle), 100 * sin(angle)); // Length of the line
    cv::line(dst, centroid - delta, centroid + delta, cv::Scalar(0, 0, 255), 2);

    // Compute the percent filled
    int area = stats.at<int>(label, cv::CC_STAT_AREA);
    double percentFilled = 100.0 * area / (rotatedRect.size.width * rotatedRect.size.height);

    // Compute the ratio of the height to the width of the bounding box
    double hwRatio = rotatedRect.size.height / rotatedRect.size.width;

    // Display the percent filled and ratio on the image
    cv::putText(dst, "Percent filled: " + std::to_string(percentFilled), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0));
    cv::putText(dst, "Height-Width Ratio: " + std::to_string(hwRatio), cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0));

    // Return the features as a vector
    return {huMoments[0], huMoments[1], huMoments[2], huMoments[3], huMoments[4], huMoments[5], huMoments[6], percentFilled, hwRatio};
}

double computeScaledEuclideanDistance(const std::vector<double> &v1, const std::vector<double> &v2, const std::vector<double> &stdev)
{
    double sum = 0;
    for (size_t i = 0; i < v1.size(); ++i)
    {
        double diff = (v1[i] - v2[i]) / stdev[i];
        sum += diff * diff; // Squaring by multiplying the value by itself
    }
    return sqrt(sum);
}

// Manual thresholding function implemented from scratch.
// This function takes a grayscale image and a threshold value.
// Pixels with intensity below the threshold become 255 (object),
// while pixels equal to or above the threshold become 0 (background).
cv::Mat threshold(const cv::Mat &src, int thresh)
{
    // Ensure the input image is a grayscale image.
    CV_Assert(src.type() == CV_8UC1);
    cv::Mat binary(src.size(), CV_8UC1);

    for (int y = 0; y < src.rows; y++)
    {
        for (int x = 0; x < src.cols; x++)
        {
            uchar pixel = src.at<uchar>(y, x);
            // If pixel intensity is less than the threshold,
            // mark it as foreground (255), otherwise as background (0).
            binary.at<uchar>(y, x) = (pixel < thresh) ? 255 : 0;
        }
    }

    return binary;
}

cv::Mat erode(const cv::Mat &src, int erosion_size)
{
    // Ensure the input is a binary image where foreground pixels are 255.
    CV_Assert(src.type() == CV_8UC1);
    cv::Mat dst = cv::Mat::zeros(src.size(), src.type());

    int height = src.rows;
    int width = src.cols;
    int kernelRadius = erosion_size; // radius in each direction

    // Loop through every pixel in the input image.
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            bool pixelShouldBeForeground = true;

            // Examine the neighborhood defined by the rectangular structuring element.
            for (int dy = -kernelRadius; dy <= kernelRadius; ++dy)
            {
                for (int dx = -kernelRadius; dx <= kernelRadius; ++dx)
                {
                    int ny = y + dy;
                    int nx = x + dx;

                    // If neighbor is out of bounds or not foreground (255),
                    // then this pixel should not remain as foreground.
                    if (ny < 0 || ny >= height || nx < 0 || nx >= width || src.at<uchar>(ny, nx) != 255)
                    {
                        pixelShouldBeForeground = false;
                        break;
                    }
                }
                if (!pixelShouldBeForeground)
                    break;
            }

            // Set the output pixel based on the flag.
            dst.at<uchar>(y, x) = pixelShouldBeForeground ? 255 : 0;
        }
    }
    return dst;
}

// ----------------------------------------------------------------------
// New helper functions for classification mode
// ----------------------------------------------------------------------

// Loads training data from CSV ("training_data.csv") and returns a vector of pairs (label, feature vector).
std::vector<std::pair<std::string, std::vector<double>>> loadTrainingData(const std::string &filename)
{
    std::vector<std::pair<std::string, std::vector<double>>> dataset;
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "[Error] Could not open training data file: " << filename << std::endl;
        return dataset;
    }
    std::string line;
    // Skip header line.
    std::getline(file, line);
    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string token;
        std::vector<std::string> tokens;
        while (std::getline(ss, token, ','))
            tokens.push_back(token);
        if (tokens.size() != 10)
            continue; // label + 9 features expected
        std::string label = tokens[0];
        std::vector<double> features;
        for (int i = 1; i < 10; i++)
            features.push_back(std::stod(tokens[i]));
        dataset.push_back({label, features});
    }
    file.close();
    return dataset;
}

// Computes the standard deviation of each feature over the training data.
// This is used to "scale" differences in the Euclidean metric.
std::vector<double> computeTrainingFeatureStdev(const std::vector<std::pair<std::string, std::vector<double>>> &dataset)
{
    if (dataset.empty())
        return std::vector<double>();
    size_t featureSize = dataset[0].second.size();
    std::vector<double> mean(featureSize, 0.0), var(featureSize, 0.0);
    size_t n = dataset.size();
    for (const auto &entry : dataset)
    {
        const std::vector<double> &features = entry.second;
        for (size_t i = 0; i < featureSize; ++i)
            mean[i] += features[i];
    }
    for (size_t i = 0; i < featureSize; ++i)
        mean[i] /= n;
    for (const auto &entry : dataset)
    {
        const std::vector<double> &features = entry.second;
        for (size_t i = 0; i < featureSize; ++i)
        {
            double diff = features[i] - mean[i];
            var[i] += diff * diff;
        }
    }
    for (size_t i = 0; i < featureSize; ++i)
        var[i] /= n;
    std::vector<double> stdev(featureSize, 0.0);
    for (size_t i = 0; i < featureSize; ++i)
    {
        stdev[i] = sqrt(var[i]);
        if (stdev[i] == 0)
            stdev[i] = 1.0;
    }
    return stdev;
}

// ----------------------------------------------------------------------

// New classifier: K-Nearest Neighbor (KNN) classifier (K > 1)
// Instead of voting, this function computes the average scaled distance for each class
// among the K nearest training examples and returns the class with the smallest average distance.
std::string classifyKNN(const std::vector<double> &candidateFeatures,
                        const std::vector<std::pair<std::string, std::vector<double>>> &trainingExamples,
                        const std::vector<double> &stdev,
                        int k)
{
    // Compute distances from the candidate features to every training example.
    std::vector<std::pair<double, std::string>> distances;
    for (const auto &entry : trainingExamples)
    {
        double d = computeScaledEuclideanDistance(candidateFeatures, entry.second, stdev);
        distances.push_back({d, entry.first});
    }
    // Sort by ascending distance.
    std::sort(distances.begin(), distances.end(), [](const auto &a, const auto &b)
              { return a.first < b.first; });

    // Use at most the first k nearest neighbors (if fewer exist, use them all)
    int effectiveK = std::min(k, static_cast<int>(distances.size()));
    std::unordered_map<std::string, double> distanceSum;
    std::unordered_map<std::string, int> count;
    for (int i = 0; i < effectiveK; ++i)
    {
        distanceSum[distances[i].second] += distances[i].first;
        count[distances[i].second] += 1;
    }

    // For each label in the K nearest neighbors, compute the average distance.
    std::string bestLabel = "Unknown";
    double bestAvgDistance = std::numeric_limits<double>::max();
    for (const auto &kv : distanceSum)
    {
        double avgDistance = kv.second / count[kv.first];
        if (avgDistance < bestAvgDistance)
        {
            bestAvgDistance = avgDistance;
            bestLabel = kv.first;
        }
    }
    return bestLabel;
}

int main(int argc, char **argv)
{
    // --- Parse Command-Line Arguments ---
    std::string imageDir = "";
    bool trainingMode = false;
    bool classifyMode = false;
    int kValue = 2; // Fixed value of k

    for (int i = 1; i < argc; ++i)
    {
        std::string arg(argv[i]);
        if (arg == "--dir" && i + 1 < argc)
        {
            imageDir = argv[i + 1];
            ++i;
        }
        else if (arg == "--train")
        {
            trainingMode = true;
        }
        else if (arg == "--classify")
        {
            classifyMode = true;
        }
    }

    if (imageDir.empty())
    {
        std::cerr << "[Error] No directory provided. Please provide a directory using the '--dir' flag." << std::endl;
        return -1;
    }

    if (trainingMode && classifyMode)
    {
        std::cerr << "[Error] Cannot use both --train and --classify modes simultaneously." << std::endl;
        return -1;
    }

    // --- Open the Training Data File if in Training Mode ---
    std::ofstream trainingFile;
    if (trainingMode)
    {
        trainingFile.open("training_data.csv", std::ios::app);
        // If the file is empty, write the header.
        if (trainingFile.tellp() == 0)
        {
            trainingFile << "Label,Hu1,Hu2,Hu3,Hu4,Hu5,Hu6,Hu7,PercentFilled,HWRatio\n";
        }
    }

    // In classification mode, load the training data.
    std::vector<std::pair<std::string, std::vector<double>>> trainingExamples;
    std::vector<double> stdev;
    if (classifyMode)
    {
        trainingExamples = loadTrainingData("training_data.csv");
        if (trainingExamples.empty())
        {
            std::cerr << "[Error] No training data found in training_data.csv" << std::endl;
            return -1;
        }
        stdev = computeTrainingFeatureStdev(trainingExamples);
    }

    namespace fs = std::filesystem;
    std::cout << "[LOG] Directory mode activated. Processing images in: " << imageDir << std::endl;

    // --- Process Each File in the Input Directory ---
    for (const auto &entry : fs::directory_iterator(imageDir))
    {
        if (!entry.is_regular_file())
            continue;

        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext != ".jpg" && ext != ".jpeg" && ext != ".png" && ext != ".bmp")
            continue;

        std::string filePath = entry.path().string();
        std::cout << "[LOG] Processing file: " << filePath << std::endl;
        cv::Mat image = cv::imread(filePath);
        if (image.empty())
        {
            std::cout << "[LOG] Could not read image: " << filePath << std::endl;
            continue;
        }

        // --- Convert to Grayscale & Optionally Pre-process ---
        cv::Mat gray;
        if (image.channels() == 3)
            cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        else
            gray = image;

        // For example, blur to uniformize regions before thresholding.
        cv::Mat blurred;
        cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);

        // --- Threshold the image (task 1) ---
        int thresholdValue = 128;
        cv::Mat thresholdedFrame = threshold(blurred, thresholdValue);

        // --- Clean the binary image using erosion (task 2) ---
        int erosion_size = 1;
        cv::Mat cleanedFrame = erode(thresholdedFrame, erosion_size);

        // --- Segment the Image into Regions (task 3) ---
        cv::Mat labels, stats, centroids;
        int numberOfLabels = cv::connectedComponentsWithStats(cleanedFrame, labels, stats, centroids);
        cv::Mat dst(cleanedFrame.size(), CV_8UC3, cv::Scalar::all(0));

        for (int label = 1; label < numberOfLabels; ++label)
        {
            int area = stats.at<int>(label, cv::CC_STAT_AREA);
            if (area > 100)
            {
                cv::Scalar color(rand() % 256, rand() % 256, rand() % 256);
                cv::Mat mask = (labels == label);
                dst.setTo(color, mask);
                std::string objectLabel = "object" + std::to_string(label);
                // --- Compute features and display overlays (task 4) ---
                std::vector<double> features = computeFeatures(labels, stats, centroids, label, dst);

                cv::Point centroidPoint(static_cast<int>(centroids.at<double>(label, 0)),
                                        static_cast<int>(centroids.at<double>(label, 1)));
                cv::putText(dst, objectLabel, centroidPoint, cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
            }
        }

        // --- Save Output Images ---
        fs::path p(filePath);
        std::string baseName = p.stem().string();
        std::string thresholdFilename = (p.parent_path() / (baseName + "_threshold" + p.extension().string())).string();
        std::string cleanedFilename = (p.parent_path() / (baseName + "_cleaned" + p.extension().string())).string();
        std::string regionFilename = (p.parent_path() / (baseName + "_region" + p.extension().string())).string();

        cv::imwrite(thresholdFilename, thresholdedFrame);
        cv::imwrite(cleanedFilename, cleanedFrame);
        cv::imwrite(regionFilename, dst);

        std::cout << "[LOG] Saved threshold image: " << thresholdFilename << std::endl;
        std::cout << "[LOG] Saved cleaned image: " << cleanedFilename << std::endl;
        std::cout << "[LOG] Saved region image: " << regionFilename << std::endl;

        // --- Training Mode: Collect Training Data (task 5) ---
        if (trainingMode)
        {
            // Identify a candidate region: prefer a region that is large and does not touch the image boundary.
            int candidateLabel = -1;
            int candidateArea = 0;
            int imgWidth = cleanedFrame.cols;
            int imgHeight = cleanedFrame.rows;
            for (int i = 1; i < numberOfLabels; ++i)
            {
                int area = stats.at<int>(i, cv::CC_STAT_AREA);
                if (area > 100)
                {
                    int x = stats.at<int>(i, cv::CC_STAT_LEFT);
                    int y = stats.at<int>(i, cv::CC_STAT_TOP);
                    int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
                    int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);
                    // Skip regions that touch the image boundary.
                    if (x == 0 || y == 0 || (x + w) >= imgWidth || (y + h) >= imgHeight)
                        continue;
                    if (area > candidateArea)
                    {
                        candidateArea = area;
                        candidateLabel = i;
                    }
                }
            }
            // If no candidate avoids the border, choose the one with maximum area.
            if (candidateLabel == -1)
            {
                for (int i = 1; i < numberOfLabels; ++i)
                {
                    int area = stats.at<int>(i, cv::CC_STAT_AREA);
                    if (area > candidateArea)
                    {
                        candidateArea = area;
                        candidateLabel = i;
                    }
                }
            }

            if (candidateLabel != -1)
            {
                // Recompute the features for the candidate region.
                std::vector<double> candidateFeatures = computeFeatures(labels, stats, centroids, candidateLabel, dst);

                // Display the result so the user can see the region.
                cv::imshow("Region", dst);
                cv::waitKey(1); // a short delay to update the window

                std::cout << "Candidate object found in " << filePath << ".\nEnter label for this object (or press Enter to skip): ";
                std::string userLabel;
                std::getline(std::cin, userLabel);

                if (!userLabel.empty())
                {
                    trainingFile << userLabel;
                    // Save each feature separated by commas.
                    for (auto feature : candidateFeatures)
                        trainingFile << "," << feature;
                    trainingFile << "\n";
                    std::cout << "[LOG] Training data saved for object: " << userLabel << std::endl;
                }
                else
                {
                    std::cout << "[LOG] No label entered. Skipping training data for this image." << std::endl;
                }
                cv::destroyWindow("Region");
            }
            else
            {
                std::cout << "[LOG] No candidate region found for training in " << filePath << std::endl;
            }
        }

        // ------------------------------------------------------------------
        // Updated classification code using the new KNN classifier (with kValue)

        if (classifyMode)
        {
            int candidateLabel = -1;
            int candidateArea = 0;
            int imgWidth = cleanedFrame.cols;
            int imgHeight = cleanedFrame.rows;
            for (int i = 1; i < numberOfLabels; ++i)
            {
                int area = stats.at<int>(i, cv::CC_STAT_AREA);
                if (area > 100)
                {
                    int x = stats.at<int>(i, cv::CC_STAT_LEFT);
                    int y = stats.at<int>(i, cv::CC_STAT_TOP);
                    int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
                    int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);
                    if (x == 0 || y == 0 || (x + w) >= imgWidth || (y + h) >= imgHeight)
                        continue;
                    if (area > candidateArea)
                    {
                        candidateArea = area;
                        candidateLabel = i;
                    }
                }
            }
            if (candidateLabel == -1)
            {
                for (int i = 1; i < numberOfLabels; ++i)
                {
                    int area = stats.at<int>(i, cv::CC_STAT_AREA);
                    if (area > candidateArea)
                    {
                        candidateArea = area;
                        candidateLabel = i;
                    }
                }
            }
            if (candidateLabel != -1)
            {
                std::vector<double> candidateFeatures = computeFeatures(labels, stats, centroids, candidateLabel, dst);
                std::string predictedLabel = classifyKNN(candidateFeatures, trainingExamples, stdev, kValue);

                // Overlay the predicted label (in blue) at the candidate region's centroid.
                cv::Point centroidPoint(static_cast<int>(centroids.at<double>(candidateLabel, 0)),
                                        static_cast<int>(centroids.at<double>(candidateLabel, 1)));
                cv::putText(dst, predictedLabel, centroidPoint, cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 0), 2);
                std::cout << "[LOG] Classified object in " << filePath << " as: " << predictedLabel << std::endl;
            }
            else
            {
                std::cout << "[LOG] No candidate region found for classification in " << filePath << std::endl;
            }
            cv::imshow("Region", dst);
            cv::waitKey(0); // wait for a key press to move to the next image
            cv::destroyWindow("Region");
        }
    }

    if (trainingMode && trainingFile.is_open())
    {
        trainingFile.close();
    }

    return 0;
}