#ifndef REGION_ANALYZER_HPP
#define REGION_ANALYZER_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <random>
#include "RegionFeatures.h"

class RegionAnalyzer
{
public:
    RegionAnalyzer(int minRegionSize = 100, int maxRegions = 5);

    // Analyze regions and return colored visualization
    cv::Mat analyzeAndVisualize(const cv::Mat &binaryImage);

    void analyzeRegion(const cv::Mat &region_map, int region_id, cv::Mat &output_image)
    {
        // Create a mask for the specified region
        cv::Mat mask = (region_map == region_id);

        // Calculate moments
        cv::Moments m = cv::moments(mask, true);
        double area = m.m00;                            // Area of the region
        cv::Point centroid(m.m10 / area, m.m01 / area); // Centroid of the region

        // Calculate bounding box
        cv::Rect bounding_box = cv::boundingRect(mask);
        double bounding_box_area = bounding_box.width * bounding_box.height;
        double percent_filled = area / bounding_box_area;

        // Calculate height/width ratio
        double height_width_ratio = static_cast<double>(bounding_box.height) / bounding_box.width;

        // Draw the bounding box and centroid on the output image
        output_image = cv::Mat::zeros(region_map.size(), CV_8UC3);
        output_image.setTo(cv::Scalar(0, 0, 0));             // Black background
        output_image.setTo(cv::Scalar(255, 255, 255), mask); // White for the region

        // Draw bounding box
        cv::rectangle(output_image, bounding_box, cv::Scalar(0, 255, 0), 2); // Green box

        // Draw centroid
        cv::circle(output_image, centroid, 5, cv::Scalar(255, 0, 0), -1); // Blue centroid

        // Display features
        std::string features = "Area: " + std::to_string(area) +
                               ", Percent Filled: " + std::to_string(percent_filled) +
                               ", Height/Width Ratio: " + std::to_string(height_width_ratio);
        cv::putText(output_image, features, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }

private:
    int minRegionSize;
    int maxRegions;
    std::vector<cv::Vec3b> colorPalette;
    std::vector<RegionFeatureData> regionFeatures;

    // Generate random color palette
    void generateColorPalette(int numColors = 256);

    // Filter and sort regions by size
    std::vector<int> filterRegions(const cv::Mat &labels, const cv::Mat &stats);

    // Create visualization of regions
    cv::Mat createVisualization(const cv::Mat &labels, const std::vector<int> &validRegions);
};

#endif // REGION_ANALYZER_HPP