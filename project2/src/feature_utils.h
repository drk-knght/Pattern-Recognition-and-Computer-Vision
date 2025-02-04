#ifndef FEATURE_UTILS_H
#define FEATURE_UTILS_H

#include <vector>
#include <opencv2/opencv.hpp>

std::vector<float> extractCenterPatch(const cv::Mat &img);
float calculateSSD(const std::vector<float> &feat1, const std::vector<float> &feat2);

std::vector<float> extractColorHistogram(const cv::Mat &img);
float calculateHistIntersection(const std::vector<float> &hist1, const std::vector<float> &hist2);

std::vector<float> extractSpatialColorHistogram(const cv::Mat &img);
float calculateSpatialHistIntersection(const std::vector<float>& hist1, const std::vector<float>& hist2);

std::vector<float> extractTextureHistogram(const cv::Mat &img);
std::vector<float> extractCombinedFeatures(const cv::Mat &img);
float calculateCombinedDistance(const std::vector<float> &feat1, const std::vector<float> &feat2);


#endif