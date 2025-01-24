/*
    Agnibha Chatterjee
    Om Agarwal
    Jan 12 2024
    CS5330- Pattern Recognition & Computer Vision
    This file is the entry
*/

#include <opencv2/opencv.hpp> // Include OpenCV header for image processing

// Function declarations for various image processing filters

// Convert a color image to grayscale
int greyscale(cv::Mat &src, cv::Mat &dst);

// Apply a sepia filter to the image
int sepia(cv::Mat &src, cv::Mat &dst);

// Apply a 5x5 Gaussian blur using the first method
int blur5x5_1(cv::Mat &src, cv::Mat &dst);

// Apply a 5x5 Gaussian blur using an alternative method
int blur5x5_2(cv::Mat &src, cv::Mat &dst);

// Apply the Sobel filter in the X direction to detect edges
int sobelX3x3(cv::Mat &src, cv::Mat &dst);

// Apply the Sobel filter in the Y direction to detect edges
int sobelY3x3(cv::Mat &src, cv::Mat &dst);

// Calculate the magnitude of the gradient from the Sobel X and Y results
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst);

// Quantize the image colors after applying a blur
int blurQuantize(cv::Mat &src, cv::Mat &dst, int levels);

// Isolate red colors in the image and combine with a grayscale version
int isolateRed(cv::Mat &src, cv::Mat dst);