#ifndef MORPHOLOGY_HPP
#define MORPHOLOGY_HPP

#include <opencv2/opencv.hpp>

class Morphology
{
public:
    static cv::Mat erode(const cv::Mat &input, int kernel_size = 3);
    static cv::Mat dilate(const cv::Mat &input, int kernel_size = 3);
    static cv::Mat opening(const cv::Mat &input, int kernel_size = 3);
    static cv::Mat closing(const cv::Mat &input, int kernel_size = 3);

private:
    static bool checkNeighborhood(const cv::Mat &input, int row, int col, int kernel_size, bool isErosion);
};

#endif // MORPHOLOGY_HPP