/*
    Agnibha Chatterjee
    Om Agarwal
    March 13 2025
    CS5330- Pattern Recognition & Computer Vision
    This header file defines the TargetDetector class for detecting and processing chessboard patterns in images using OpenCV.
*/
#ifndef TARGET_DETECTOR_H
#define TARGET_DETECTOR_H

#include <opencv2/opencv.hpp>
#include <vector>

// Class for detecting and processing chessboard patterns in images
class TargetDetector
{
public:
    // Constructor to initialize the detector with board dimensions
    TargetDetector(int boardWidth = 9, int boardHeight = 6);

    // Function to detect corners in a given image
    bool detectCorners(const cv::Mat &image, std::vector<cv::Point2f> &corners);

    // Function to draw detected corners on the image
    void drawCorners(cv::Mat &image, const std::vector<cv::Point2f> &corners);

    // Function to refine the detected corners for better accuracy
    void refineCorners(const cv::Mat &image, std::vector<cv::Point2f> &corners);

private:
    int mBoardWidth;     // Width of the chessboard (number of internal corners)
    int mBoardHeight;    // Height of the chessboard (number of internal corners)
    cv::Size mBoardSize; // Size of the chessboard
};

#endif