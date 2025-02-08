/*
    Agnibha Chatterjee
    Om Agarwal
    Feb 8 2025
    CS5330- Pattern Recognition & Computer Vision
    This file extracts Local Binary Pattern (LBP) features from an input image and outputs a normalized histogram.
*/

#include <opencv2/opencv.hpp>
#include "feature_utils.h"

std::vector<float> extractLBPFeatures(const cv::Mat &img)
{
    // Convert the input image to grayscale if necessary.
    cv::Mat gray;
    if (img.channels() == 3)
    {
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    }
    else
    {
        gray = img;
    }

    int rows = gray.rows;
    int cols = gray.cols;

    // Create an LBP image (excluding border pixels).
    cv::Mat lbpImage = cv::Mat::zeros(rows - 2, cols - 2, CV_8UC1);

    // Compute the LBP code for each pixel (ignoring the border).
    // Order: top-left, top, top-right, right, bottom-right, bottom, bottom-left, left.
    for (int i = 1; i < rows - 1; i++)
    {
        for (int j = 1; j < cols - 1; j++)
        {
            uchar center = gray.at<uchar>(i, j);
            unsigned char code = 0;
            code |= (gray.at<uchar>(i - 1, j - 1) >= center) << 7;
            code |= (gray.at<uchar>(i - 1, j) >= center) << 6;
            code |= (gray.at<uchar>(i - 1, j + 1) >= center) << 5;
            code |= (gray.at<uchar>(i, j + 1) >= center) << 4;
            code |= (gray.at<uchar>(i + 1, j + 1) >= center) << 3;
            code |= (gray.at<uchar>(i + 1, j) >= center) << 2;
            code |= (gray.at<uchar>(i + 1, j - 1) >= center) << 1;
            code |= (gray.at<uchar>(i, j - 1) >= center) << 0;
            lbpImage.at<uchar>(i - 1, j - 1) = code;
        }
    }

    // Build a histogram with 256 bins.
    const int histSize = 256;
    std::vector<float> hist(histSize, 0.0f);
    for (int i = 0; i < lbpImage.rows; i++)
    {
        for (int j = 0; j < lbpImage.cols; j++)
        {
            uchar code = lbpImage.at<uchar>(i, j);
            hist[code] += 1.0f;
        }
    }

    // Normalize the histogram.
    float totalPixels = static_cast<float>(lbpImage.rows * lbpImage.cols);
    for (int i = 0; i < histSize; i++)
    {
        hist[i] /= totalPixels;
    }

    return hist;
}