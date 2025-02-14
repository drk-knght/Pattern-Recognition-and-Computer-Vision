#ifndef REGION_ANALYZER_HPP
#define REGION_ANALYZER_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <random>

class RegionAnalyzer
{
public:
    RegionAnalyzer(int minRegionSize = 100, int maxRegions = 5);

    // Analyze regions and return colored visualization
    cv::Mat analyzeAndVisualize(const cv::Mat &binaryImage);

private:
    int minRegionSize;
    int maxRegions;
    std::vector<cv::Vec3b> colorPalette;

    // Generate random color palette
    void generateColorPalette(int numColors = 256);

    // Filter and sort regions by size
    std::vector<int> filterRegions(const cv::Mat &labels, const cv::Mat &stats);

    // Create visualization of regions
    cv::Mat createVisualization(const cv::Mat &labels, const std::vector<int> &validRegions);
};

#endif // REGION_ANALYZER_HPP