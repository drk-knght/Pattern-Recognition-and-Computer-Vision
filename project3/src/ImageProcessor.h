#ifndef IMAGE_PROCESSOR_HPP
#define IMAGE_PROCESSOR_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <random>
#include "RegionAnalyzer.h"
class ImageProcessor
{
public:
    ImageProcessor();

    // Process single frame
    cv::Mat preprocess_image(const cv::Mat &frame);
    int get_dynamic_threshold(const cv::Mat &gray_image);

    // Process and display results
    void process_frame(const cv::Mat &frame, cv::Mat &threshold_frame);

    // Window management
    void create_windows();
    void destroy_windows();

    cv::Mat get_threshold_frame(); // Declaration

private:
    // Window names
    const std::string WINDOW_ORIGINAL = "Original";
    const std::string WINDOW_PROCESSED = "Processed";
    RegionAnalyzer regionAnalyzer;
    cv::Mat regions_frame;

    // Random number generation
    std::random_device rd;
    std::mt19937 gen;
};

#endif // IMAGE_PROCESSOR_HPP