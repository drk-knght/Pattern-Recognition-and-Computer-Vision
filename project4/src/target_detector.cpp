#include "target_detector.h"

TargetDetector::TargetDetector(int boardWidth, int boardHeight)
    : mBoardWidth(boardWidth), mBoardHeight(boardHeight), mBoardSize(boardWidth, boardHeight)
{
}

bool TargetDetector::detectCorners(const cv::Mat &image, std::vector<cv::Point2f> &corners)
{
    bool found = cv::findChessboardCorners(image, mBoardSize, corners,
                                           cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_FAST_CHECK);

    if (found)
    {
        refineCorners(image, corners);
    }
    return found;
}

void TargetDetector::refineCorners(const cv::Mat &image, std::vector<cv::Point2f> &corners)
{
    cv::Mat gray;
    if (image.channels() == 3)
    {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    }
    else
    {
        gray = image.clone();
    }

    cv::Size winSize(11, 11);
    cv::Size zeroZone(-1, -1);
    cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.001);

    cv::cornerSubPix(gray, corners, winSize, zeroZone, criteria);
}

void TargetDetector::drawCorners(cv::Mat &image, const std::vector<cv::Point2f> &corners)
{
    cv::drawChessboardCorners(image, mBoardSize, corners, !corners.empty());
}