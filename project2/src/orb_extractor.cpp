#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include "feature_utils.h"

// This function uses OpenCV's built-in ORB detector to compute descriptors,
// then computes the mean descriptor (aggregating across keypoints) to produce
// a fixed-length feature vector.
std::vector<float> extractORBFeatures(const cv::Mat &img)
{
    int nFeatures = 500;
    cv::Ptr<cv::ORB> orb = cv::ORB::create(nFeatures);
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    orb->detectAndCompute(img, cv::noArray(), keypoints, descriptors);

    // If no descriptors were found, return an empty feature vector.
    if (descriptors.empty())
    {
        return std::vector<float>();
    }

    // Convert descriptors from CV_8U to CV_32F so we can average them.
    cv::Mat descriptorsFloat;
    descriptors.convertTo(descriptorsFloat, CV_32F);

    // Compute the average (mean) descriptor over all keypoints.
    cv::Mat meanDescriptor;
    cv::reduce(descriptorsFloat, meanDescriptor, 0, cv::REDUCE_AVG);

    // Convert the mean descriptor (a 1 x D matrix) into a std::vector<float>
    std::vector<float> feature(meanDescriptor.cols);
    for (int i = 0; i < meanDescriptor.cols; i++)
    {
        feature[i] = meanDescriptor.at<float>(0, i);
    }
    return feature;
}

#ifdef ORB_EXTRACTOR_MAIN
// Demo main function to visualize ORB keypoints and print the feature vector size.
// Compile with -DORB_EXTRACTOR_MAIN to enable.
int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return -1;
    }

    // Load the image.
    std::string imagePath = argv[1];
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (image.empty())
    {
        std::cerr << "Error: Unable to load image: " << imagePath << std::endl;
        return -1;
    }

    // Extract ORB features.
    std::vector<float> orbFeature = extractORBFeatures(image);
    std::cout << "Extracted ORB feature vector size: " << orbFeature.size() << std::endl;

    // For visualization: detect keypoints using ORB.
    int nFeatures = 500;
    cv::Ptr<cv::ORB> orb = cv::ORB::create(nFeatures);
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    orb->detectAndCompute(image, cv::noArray(), keypoints, descriptors);

    cv::Mat output;
    cv::drawKeypoints(image, keypoints, output, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::imshow("ORB Keypoints", output);
    cv::waitKey(0);

    return 0;
}
#endif