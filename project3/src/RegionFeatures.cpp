#include "RegionFeatures.h"

RegionFeatureData RegionFeatures::computeFeatures(const cv::Mat& labeledImage, int regionLabel) {
    RegionFeatureData features;
    
    // Extract region mask
    cv::Mat mask = extractRegionMask(labeledImage, regionLabel);
    
    // Calculate moments
    cv::Moments moments = cv::moments(mask, true);
    
    // Calculate center of mass
    features.center = cv::Point2f(
        moments.m10 / moments.m00,
        moments.m01 / moments.m00
    );
    
    // Calculate orientation (angle of least central moment)
    double mu11 = moments.mu11;
    double mu20 = moments.mu20;
    double mu02 = moments.mu02;
    features.orientation = 0.5 * atan2(2 * mu11, mu20 - mu02);
    
    // Find contours for the region
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    if (!contours.empty()) {
        // Calculate oriented bounding box
        features.orientedBoundingBox = cv::minAreaRect(contours[0]);
        
        // Get the four corners of the oriented bounding box
        cv::Point2f boxPoints[4];
        features.orientedBoundingBox.points(boxPoints);
        features.boxPoints = std::vector<cv::Point2f>(boxPoints, boxPoints + 4);
        
        // Calculate area and perimeter
        features.area = cv::contourArea(contours[0]);
        features.perimeter = cv::arcLength(contours[0], true);
        
        // Calculate fill ratio (area / bounding box area)
        features.fillRatio = features.area / (features.orientedBoundingBox.size.width * 
                                            features.orientedBoundingBox.size.height);
        
        // Calculate aspect ratio
        features.aspectRatio = std::max(features.orientedBoundingBox.size.width, 
                                      features.orientedBoundingBox.size.height) /
                              std::min(features.orientedBoundingBox.size.width, 
                                     features.orientedBoundingBox.size.height);
    }
    
    return features;
}

void RegionFeatures::drawFeatures(cv::Mat& image, const RegionFeatureData& features) {
    // Draw oriented bounding box
    for (size_t i = 0; i < 4; i++) {
        cv::line(image, features.boxPoints[i], 
                features.boxPoints[(i + 1) % 4], 
                cv::Scalar(0, 255, 0), 2);
    }
    
    // Draw center point
    cv::circle(image, features.center, 5, cv::Scalar(0, 0, 255), -1);
    
    // Draw orientation line
    cv::Point2f endPoint(
        features.center.x + 50 * cos(features.orientation),
        features.center.y + 50 * sin(features.orientation)
    );
    cv::line(image, features.center, endPoint, cv::Scalar(255, 0, 0), 2);
    
    // Draw feature values
    std::string text = cv::format("AR: %.2f FR: %.2f", 
                                 features.aspectRatio, 
                                 features.fillRatio);
    cv::putText(image, text, 
                cv::Point(features.center.x - 50, features.center.y - 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
}

cv::Mat RegionFeatures::extractRegionMask(const cv::Mat& labeledImage, int regionLabel) {
    cv::Mat mask = cv::Mat::zeros(labeledImage.size(), CV_8UC1);
    mask = labeledImage == regionLabel;
    return mask;
}

std::vector<cv::Point> RegionFeatures::getRegionPoints(const cv::Mat& mask) {
    std::vector<cv::Point> points;
    for (int y = 0; y < mask.rows; y++) {
        for (int x = 0; x < mask.cols; x++) {
            if (mask.at<uchar>(y, x) > 0) {
                points.push_back(cv::Point(x, y));
            }
        }
    }
    return points;
}