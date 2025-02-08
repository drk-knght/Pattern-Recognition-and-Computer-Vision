/*
    Agnibha Chatterjee
    Om Agarwal
    Feb 8 2025
    CS5330- Pattern Recognition & Computer Vision
    This file extracts ORB features from an input image and computes a mean descriptor.
*/

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include "feature_utils.h"

std::vector<float> extractORBFeatures(const cv::Mat &img)
{
    int nFeatures = 500;
    cv::Ptr<cv::ORB> orb = cv::ORB::create(nFeatures);
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    orb->detectAndCompute(img, cv::noArray(), keypoints, descriptors);

    if (descriptors.empty())
    {
        return std::vector<float>();
    }

    cv::Mat descriptorsFloat;
    descriptors.convertTo(descriptorsFloat, CV_32F);

    cv::Mat meanDescriptor;
    cv::reduce(descriptorsFloat, meanDescriptor, 0, cv::REDUCE_AVG);

    std::vector<float> feature(meanDescriptor.cols);
    for (int i = 0; i < meanDescriptor.cols; i++)
    {
        feature[i] = meanDescriptor.at<float>(0, i);
    }
    return feature;
}