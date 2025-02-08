/*
    Agnibha Chatterjee
    Om Agarwal
    Feb 8 2025
    CS5330- Pattern Recognition & Computer Vision
    This file declares utility functions for feature extraction and distance calculations used in the image matching pipeline.
*/

#ifndef FEATURE_UTILS_H
#define FEATURE_UTILS_H

#include <vector>
#include <opencv2/opencv.hpp>

std::vector<float> extractCenterPatch(const cv::Mat &img);
float calculateSSD(const std::vector<float> &feat1, const std::vector<float> &feat2);

std::vector<float> extractColorHistogram(const cv::Mat &img);
float calculateHistIntersection(const std::vector<float> &hist1, const std::vector<float> &hist2);

std::vector<float> extractSpatialColorHistogram(const cv::Mat &img);
float calculateSpatialHistIntersection(const std::vector<float> &hist1, const std::vector<float> &hist2);

std::vector<float> extractTextureHistogram(const cv::Mat &img);
std::vector<float> extractCombinedFeatures(const cv::Mat &img);
float calculateCombinedDistance(const std::vector<float> &feat1, const std::vector<float> &feat2);

// New ORB feature extraction function declaration.
std::vector<float> extractORBFeatures(const cv::Mat &img);

// New LBP feature extraction function declaration.
std::vector<float> extractLBPFeatures(const cv::Mat &img);

std::vector<float> extractSSIMFeatures(const cv::Mat &img);

// Add this declaration
std::vector<float> extractColorSpatialVariance(const cv::Mat &img);
float calculateSpatialVarianceDistance(const std::vector<float> &feat1, const std::vector<float> &feat2);

std::vector<float> extractCombinedDnnSpatialVarianceFeatures(const cv::Mat &img);

#endif