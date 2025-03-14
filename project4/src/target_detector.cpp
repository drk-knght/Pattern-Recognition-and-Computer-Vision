/*
    Agnibha Chatterjee
    Om Agarwal
    March 13 2025
    CS5330- Pattern Recognition & Computer Vision
    This file implements a target detection system using OpenCV to detect, refine, and draw corners of a chessboard pattern in images.
*/
#include "target_detector.h"

// Constructor for TargetDetector class
TargetDetector::TargetDetector(int boardWidth, int boardHeight)
    : mBoardWidth(boardWidth), mBoardHeight(boardHeight), mBoardSize(boardWidth, boardHeight)
{
    // Initialize board size with given width and height
}

// Function to detect corners in a chessboard pattern within an image
bool TargetDetector::detectCorners(const cv::Mat &image, std::vector<cv::Point2f> &corners)
{
    // Attempt to find chessboard corners in the image
    bool found = cv::findChessboardCorners(image, mBoardSize, corners,
                                           cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_FAST_CHECK);

    // If corners are found, refine their positions
    if (found)
    {
        refineCorners(image, corners);
    }
    return found; // Return whether corners were found
}

// Function to refine the detected corners for better accuracy
void TargetDetector::refineCorners(const cv::Mat &image, std::vector<cv::Point2f> &corners)
{
    cv::Mat gray;
    // Convert image to grayscale if it is in color
    if (image.channels() == 3)
    {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    }
    else
    {
        gray = image.clone(); // Clone the image if already grayscale
    }

    // Parameters for corner refinement
    cv::Size winSize(11, 11);                                                              // Search window size
    cv::Size zeroZone(-1, -1);                                                             // Dead zone in the middle of the search zone
    cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.001); // Termination criteria

    // Refine corner locations
    cv::cornerSubPix(gray, corners, winSize, zeroZone, criteria);
}

// Function to draw the detected corners on the image
void TargetDetector::drawCorners(cv::Mat &image, const std::vector<cv::Point2f> &corners)
{
    // Draw the corners on the image
    cv::drawChessboardCorners(image, mBoardSize, corners, !corners.empty());
}