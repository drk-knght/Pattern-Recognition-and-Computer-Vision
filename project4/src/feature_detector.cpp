/*
    Agnibha Chatterjee
    Om Agarwal
    March 13 2025
    CS5330- Pattern Recognition & Computer Vision
    This file implements a feature detection system using OpenCV to detect and visualize features using Harris and FAST methods.
*/
#include <opencv2/opencv.hpp>
#include <iostream>

// Class to handle feature detection using Harris and FAST methods
class FeatureDetector
{
private:
    // Harris corner parameters
    int blockSize = 2;      // Size of neighborhood considered for corner detection
    int apertureSize = 3;   // Aperture parameter for the Sobel operator
    double k = 0.04;        // Harris detector free parameter
    double threshold = 150; // Threshold for detecting corners

    // FAST parameters
    int fastThreshold = 20;        // Threshold for FAST corner detection
    bool nonmaxSuppression = true; // Whether to apply non-maximum suppression

public:
    // Enum to select the type of detector
    enum DetectorType
    {
        HARRIS,
        FAST
    };
    DetectorType currentDetector = HARRIS; // Default detector is Harris

    // Function to detect and draw features on the frame
    void detectAndDrawFeatures(cv::Mat &frame)
    {
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY); // Convert frame to grayscale

        // Detect features based on the current detector type
        if (currentDetector == HARRIS)
        {
            detectHarrisCorners(gray, frame);
        }
        else
        {
            detectFastFeatures(gray, frame);
        }

        // Display the current detector and parameters on the frame
        std::string detectorName = (currentDetector == HARRIS) ? "Harris" : "FAST";
        cv::putText(frame, "Detector: " + detectorName,
                    cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    cv::Scalar(0, 255, 0), 2);

        if (currentDetector == HARRIS)
        {
            cv::putText(frame, "Threshold: " + std::to_string(int(threshold)),
                        cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                        cv::Scalar(0, 255, 0), 2);
            cv::putText(frame, "Block Size: " + std::to_string(blockSize),
                        cv::Point(10, 90), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                        cv::Scalar(0, 255, 0), 2);
        }
        else
        {
            cv::putText(frame, "Threshold: " + std::to_string(fastThreshold),
                        cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                        cv::Scalar(0, 255, 0), 2);
            cv::putText(frame, "NonMaxSuppression: " + std::to_string(nonmaxSuppression),
                        cv::Point(10, 90), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                        cv::Scalar(0, 255, 0), 2);
        }
    }

    // Function to detect Harris corners
    void detectHarrisCorners(const cv::Mat &gray, cv::Mat &output)
    {
        cv::Mat dst = cv::Mat::zeros(gray.size(), CV_32FC1);

        // Detect Harris corners
        cv::cornerHarris(gray, dst, blockSize, apertureSize, k);

        // Normalize and scale the result
        cv::Mat dst_norm;
        cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());

        // Draw circles around detected corners
        for (int i = 0; i < dst_norm.rows; i++)
        {
            for (int j = 0; j < dst_norm.cols; j++)
            {
                if ((int)dst_norm.at<float>(i, j) > threshold)
                {
                    cv::circle(output, cv::Point(j, i), 5, cv::Scalar(0, 0, 255), 2);
                }
            }
        }
    }

    // Function to detect FAST features
    void detectFastFeatures(const cv::Mat &gray, cv::Mat &output)
    {
        std::vector<cv::KeyPoint> keypoints;
        cv::FAST(gray, keypoints, fastThreshold, nonmaxSuppression);

        // Draw circles around detected keypoints
        for (const auto &kp : keypoints)
        {
            cv::circle(output, kp.pt, 5, cv::Scalar(255, 0, 0), 2);
        }
    }

    // Function to adjust parameters based on user input
    void adjustParameters(char key)
    {
        switch (key)
        {
        case 'd': // Switch between Harris and FAST detectors
            currentDetector = (currentDetector == HARRIS) ? FAST : HARRIS;
            break;

        // Adjust Harris parameters
        case 'q':
            threshold = std::max(0.0, threshold - 10); // Decrease threshold
            break;
        case 'w':
            threshold += 10; // Increase threshold
            break;
        case 'a':
            blockSize = std::max(2, blockSize - 1); // Decrease block size
            break;
        case 's':
            blockSize++; // Increase block size
            break;

        // Adjust FAST parameters
        case 'z':
            fastThreshold = std::max(1, fastThreshold - 1); // Decrease FAST threshold
            break;
        case 'x':
            fastThreshold++; // Increase FAST threshold
            break;
        case 'n':
            nonmaxSuppression = !nonmaxSuppression; // Toggle non-max suppression
            break;
        }
    }
};

int main()
{
    cv::VideoCapture cap(0); // Open the default camera
    if (!cap.isOpened())
    {
        std::cerr << "Error: Could not open camera." << std::endl;
        return -1;
    }

    FeatureDetector detector; // Create a FeatureDetector object

    // Display control instructions
    std::cout << "\nControls:" << std::endl;
    std::cout << "  q/w - Decrease/Increase threshold" << std::endl;
    std::cout << "  a/s - Decrease/Increase block size (Harris)" << std::endl;
    std::cout << "  z/x - Decrease/Increase FAST threshold" << std::endl;
    std::cout << "  n   - Toggle non-max suppression (FAST)" << std::endl;
    std::cout << "  d   - Switch between Harris and FAST" << std::endl;
    std::cout << "  ESC - Quit" << std::endl;

    while (true)
    {
        cv::Mat frame;
        cap >> frame; // Capture a new frame
        if (frame.empty())
            break;

        detector.detectAndDrawFeatures(frame); // Detect and draw features

        cv::imshow("Feature Detection", frame); // Display the frame

        char key = cv::waitKey(1); // Wait for a key press
        if (key == 27)
            break;                      // Exit on ESC key
        detector.adjustParameters(key); // Adjust parameters based on key press
    }

    cap.release();           // Release the camera
    cv::destroyAllWindows(); // Close all OpenCV windows
    return 0;
}