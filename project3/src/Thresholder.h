#ifndef THRESHOLDER_HPP
#define THRESHOLDER_HPP

#include <opencv2/opencv.hpp>

class Thresholder {
public:
    // Apply binary inverse thresholding
    static cv::Mat apply(const cv::Mat& input, int threshold_value);

private:
    // Helper method to get output pixel value
    static uchar get_threshold_value(uchar input_pixel, int threshold_value);
};

#endif // THRESHOLDER_HPP