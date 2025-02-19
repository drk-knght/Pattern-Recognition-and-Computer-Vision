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

    // Process a single frame
    cv::Mat preprocess_image(const cv::Mat &frame);
    int get_dynamic_threshold(const cv::Mat &gray_image);

    // Process and display results
    void process_frame(const cv::Mat &frame, cv::Mat &threshold_frame);

    // Window management
    void create_windows();
    void destroy_windows();

    // Added for training data collection:
    // Returns the binary (thresholded and morphologically filtered) image used for region analysis.
    cv::Mat get_threshold_frame();
    // Extracts features from the binary image and appends them (with a label provided by the user) to a CSV file.
    void collect_training_data(const cv::Mat &binaryFrame);

private:
    // Window names
    const std::string WINDOW_ORIGINAL = "Original";
    const std::string WINDOW_PROCESSED = "Processed";
    RegionAnalyzer regionAnalyzer;
    cv::Mat regions_frame; // Will store the binary image used for region analysis

    // Random number generation
    std::random_device rd;
    std::mt19937 gen;
};

#endif // IMAGE_PROCESSOR_HPP