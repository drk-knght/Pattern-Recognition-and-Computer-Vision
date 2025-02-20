/*
CS 5330 Pattern Recognition & Computer Vision

Project 3 - 2-D Object Recognition

Purpose : To implement a real-time 2D object recognition system using OpenCV and C++.

Description: The following program implements functionalities to perform the following:
1. Thresholding
2. Cleaning the Image (ERODE)
3. Segmentation
4. Feature Extraction
5. Classification
6. Training
7. Classfication
8. Second Classification using K-nearest neighbours with value = 2
  => Present in Threshold2.cpp

Basics of the program:
1. User presses 't' to display the thresholded video
2. User presses 'c' to display the cleaned video
3. User presses 'r' to display the regions
4. User presses 'n' to prompt for a label and store the features
5. User presses 'i' to end the training phase and begin the classification phase
6. User presses 'q' to quit the program

Extensions Implemented:
1. Classifies multiple objects in the frame
2. Displays the label "Unknown" if the object is not in the training data
3. Classifies the "Unknown" object and stores it in the training data
4. User Interaction
5. No limit on the number of objects in the training data

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

    // Draw the orientation
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

int main(int argc, char **argv)
{
    // Open the video capture
    cv::VideoCapture cap(0); // open the default camera
    if (!cap.isOpened())     // check if we succeeded
        return -1;

    cv::namedWindow("Original Video", 1);
    cv::namedWindow("Thresholded Video", 1);
    cv::namedWindow("Cleaned Video", 1);
    cv::namedWindow("Regions", 1);

    bool displayThresholded = false;
    bool displayCleaned = false;
    bool displayRegions = false;

    std::map<cv::Point2d, cv::Scalar, Point2dComparator> previousCentroids;

    char key = 0;
    bool isTraining = true;

    while (key != 'q')
    {
        cv::Mat frame;
        cap >> frame; // get a new frame from camera

        // Display the original video
        cv::imshow("Original Video", frame);

        // Convert the captured frame to grayscale.
        cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);

        // Apply the manual thresholding function from scratch.
        // Here, we use a constant threshold value (e.g. 128).
        // Since the objects are dark, pixels below 128 become foreground (255).
        int thresholdValue = 128; // Adjust this based on your scene conditions.
        cv::Mat thresholdedFrame = threshold(frame, thresholdValue);

        // Instead of using cv::threshold and cv::bitwise_not, we now get a binary image directly.
        // The rest of the code can then work on thresholdedFrame, for example applying morphological operations.

        // Create a structuring element and apply erosion for cleaning up the binary image.
        int erosion_size = 1;
        cv::Mat cleanedFrame = erode(thresholdedFrame, erosion_size);

        // Perform connected components analysis
        cv::Mat labels, stats, centroids;
        int numberOfLabels = cv::connectedComponentsWithStats(cleanedFrame, labels, stats, centroids);

        cv::Mat dst(cleanedFrame.size(), CV_8UC3);
        dst = cv::Scalar::all(0);

        std::map<cv::Point2d, cv::Scalar, Point2dComparator> currentCentroids;

        // Display the regions and process each region/object
        for (int label = 1; label < numberOfLabels; ++label)
        {
            int area = stats.at<int>(label, cv::CC_STAT_AREA);
            if (area > 100)
            { // Ignore regions that are too small
                cv::Point2d centroid(centroids.at<double>(label, 0), centroids.at<double>(label, 1));
                cv::Scalar color;
                if (previousCentroids.count(centroid) > 0)
                {
                    color = previousCentroids[centroid];
                }
                else
                {
                    color = cv::Scalar(rand() % 256, rand() % 256, rand() % 256);
                }
                currentCentroids[centroid] = color;
                cv::Mat mask = (labels == label);
                dst.setTo(color, mask);

                // Compute the features for the region
                std::vector<double> features = computeFeatures(labels, stats, centroids, label, dst);

                if (isTraining)
                {
                    // When 'n' is pressed, prompt for a label and store the features.
                    if (key == 'n')
                    {
                        std::cout << "[LOG] 'N' pressed: Capturing training data for region (label " << label << ")." << std::endl;
                        std::string objectLabel;
                        std::cout << "Enter label for current object: ";
                        std::cin >> objectLabel;
                        trainingData[objectLabel] = features;
                        std::cout << "[LOG] Training data stored for object \"" << objectLabel << "\"." << std::endl;
                    }
                }
                else
                {
                    // Compute the standard deviation for each feature
                    std::vector<double> stdev(trainingData.begin()->second.size(), 0);
                    for (const auto &pair : trainingData)
                    {
                        for (size_t i = 0; i < pair.second.size(); ++i)
                        {
                            stdev[i] += pow(pair.second[i], 2);
                        }
                    }
                    for (double &val : stdev)
                    {
                        val = sqrt(val / trainingData.size());
                    }

                    // Classify the object
                    std::string objectLabel;
                    double minDistance = std::numeric_limits<double>::max();
                    for (const auto &pair : trainingData)
                    {
                        double distance = computeScaledEuclideanDistance(features, pair.second, stdev);
                        if (distance < minDistance)
                        {
                            minDistance = distance;
                            objectLabel = pair.first;
                        }
                    }
                    if (minDistance > 100)
                    {
                        objectLabel = "unknown";
                        std::string newLabel = "object" + std::to_string(trainingData.size());
                        trainingData[newLabel] = features;
                        objectLabel = newLabel;
                    }

                    // Display the label on the image
                    cv::putText(dst, objectLabel, cv::Point(centroids.at<double>(label, 0), centroids.at<double>(label, 1)), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0));
                }
            }
        }

        previousCentroids = currentCentroids;

        // Display the thresholded video if 't' is pressed (toggle display)
        if (key == 't')
        {
            displayThresholded = !displayThresholded;
            std::cout << "[LOG] 'T' pressed: Toggling Thresholded Video display. Now displayThresholded is " << (displayThresholded ? "ON" : "OFF") << std::endl;
        }

        // Display the cleaned video if 'c' is pressed (toggle display)
        if (key == 'c')
        {
            displayCleaned = !displayCleaned;
            std::cout << "[LOG] 'C' pressed: Toggling Cleaned Video display. Now displayCleaned is " << (displayCleaned ? "ON" : "OFF") << std::endl;
        }

        // Display the regions if 'r' is pressed (toggle display)
        if (key == 'r')
        {
            displayRegions = !displayRegions;
            std::cout << "[LOG] 'R' pressed: Toggling Regions display. Now displayRegions is " << (displayRegions ? "ON" : "OFF") << std::endl;
        }

        // End the training phase and begin classification when 'i' is pressed.
        if (key == 'i')
        {
            isTraining = false;
            std::cout << "[LOG] 'I' pressed: Ending training phase, starting classification." << std::endl;
        }

        // Show the various windows if toggled on
        if (displayThresholded)
        {
            cv::imshow("Thresholded Video", thresholdedFrame);
        }

        if (displayCleaned)
        {
            cv::imshow("Cleaned Video", cleanedFrame);
        }

        if (displayRegions)
        {
            cv::imshow("Regions", dst);
        }

        // Wait for key press and log if 'q' is pressed to quit
        int k = cv::waitKey(30);
        if (k == 'q')
        {
            std::cout << "[LOG] 'Q' pressed: Exiting the program." << std::endl;
        }
        key = k;
    }

    // After the training session, write the training data to a file
    std::ofstream file("trainingData.txt");
    for (const auto &pair : trainingData)
    {
        file << pair.first << " ";
        for (double feature : pair.second)
        {
            file << feature << " ";
        }
        file << "\n";
    }

    return 0;
}