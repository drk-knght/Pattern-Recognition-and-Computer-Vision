#include "RegionAnalyzer.h"

RegionAnalyzer::RegionAnalyzer(int minRegionSize, int maxRegions)
    : minRegionSize(minRegionSize), maxRegions(maxRegions)
{
    generateColorPalette();
}

void RegionAnalyzer::generateColorPalette(int numColors)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(50, 255); // Avoid too dark colors

    colorPalette.clear();
    for (int i = 0; i < numColors; i++)
    {
        colorPalette.push_back(cv::Vec3b(dis(gen), dis(gen), dis(gen)));
    }
}

std::vector<int> RegionAnalyzer::filterRegions(const cv::Mat &labels, const cv::Mat &stats)
{
    std::vector<std::pair<int, int>> regions; // (area, label) pairs

    // Skip background (label 0)
    for (int label = 1; label < stats.rows; label++)
    {
        int area = stats.at<int>(label, cv::CC_STAT_AREA);
        if (area >= minRegionSize)
        {
            regions.push_back({area, label});
        }
    }

    // Sort regions by area in descending order
    std::sort(regions.begin(), regions.end(),
              [](const auto &a, const auto &b)
              { return a.first > b.first; });

    // Take only the largest maxRegions
    std::vector<int> validRegions;
    for (int i = 0; i < std::min(maxRegions, (int)regions.size()); i++)
    {
        validRegions.push_back(regions[i].second);
    }

    return validRegions;
}

cv::Mat RegionAnalyzer::createVisualization(const cv::Mat &labels,
                                            const std::vector<int> &validRegions)
{
    cv::Mat visualization = cv::Mat::zeros(labels.size(), CV_8UC3);

    for (int i = 0; i < labels.rows; i++)
    {
        for (int j = 0; j < labels.cols; j++)
        {
            int label = labels.at<int>(i, j);
            if (std::find(validRegions.begin(), validRegions.end(), label) != validRegions.end())
            {
                visualization.at<cv::Vec3b>(i, j) = colorPalette[label % colorPalette.size()];
            }
        }
    }

    return visualization;
}

cv::Mat RegionAnalyzer::analyzeAndVisualize(const cv::Mat &binaryImage)
{
    cv::Mat labels, stats, centroids;
    int numLabels = cv::connectedComponentsWithStats(
        binaryImage, labels, stats, centroids, 8, CV_32S);

    std::vector<int> validRegions = filterRegions(labels, stats);
    return createVisualization(labels, validRegions);
}