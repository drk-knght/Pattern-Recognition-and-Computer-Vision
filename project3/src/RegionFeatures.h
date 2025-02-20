#ifndef REGION_FEATURES_H
#define REGION_FEATURES_H

#include <opencv2/opencv.hpp>
#include <vector>

struct RegionFeatureData {
    double area;
    double perimeter;
    double fillRatio;
    double aspectRatio;
    cv::Point2f center;
    double orientation;
    cv::RotatedRect orientedBoundingBox;
    std::vector<cv::Point2f> boxPoints;
};

class RegionFeatures {
public:
    static RegionFeatureData computeFeatures(const cv::Mat& labeledImage, int regionLabel);
    static void drawFeatures(cv::Mat& image, const RegionFeatureData& features);

private:
    static cv::Mat extractRegionMask(const cv::Mat& labeledImage, int regionLabel);
    static std::vector<cv::Point> getRegionPoints(const cv::Mat& mask);
};

#endif // REGION_FEATURES_H