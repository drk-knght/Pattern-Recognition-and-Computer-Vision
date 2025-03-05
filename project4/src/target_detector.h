#ifndef TARGET_DETECTOR_H
#define TARGET_DETECTOR_H

#include <opencv2/opencv.hpp>
#include <vector>

class TargetDetector {
public:
    TargetDetector(int boardWidth = 9, int boardHeight = 6);
    bool detectCorners(const cv::Mat& image, std::vector<cv::Point2f>& corners);
    void drawCorners(cv::Mat& image, const std::vector<cv::Point2f>& corners);
    void refineCorners(const cv::Mat& image, std::vector<cv::Point2f>& corners);

private:
    int mBoardWidth;
    int mBoardHeight;
    cv::Size mBoardSize;
};

#endif