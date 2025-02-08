/*
    Agnibha Chatterjee
    Om Agarwal
    Feb 8 2025
    CS5330- Pattern Recognition & Computer Vision
    This file is the entry point for question 7 of the assignment.
*/

#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>

std::vector<float> extractSSIMFeatures(const cv::Mat &img)
{
    // Convert the input image to grayscale if it is not already.
    cv::Mat gray;
    if (img.channels() == 3)
    {
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    }
    else
    {
        gray = img;
    }

    // Convert the image to float (CV_32F)
    cv::Mat I;
    gray.convertTo(I, CV_32F);

    // Compute a Gaussian-blurred version of the image.
    cv::Mat I_blur;
    cv::GaussianBlur(I, I_blur, cv::Size(11, 11), 1.5);

    // Calculate local means using Gaussian filtering.
    cv::Mat mu1, mu2;
    cv::GaussianBlur(I, mu1, cv::Size(11, 11), 1.5);
    cv::GaussianBlur(I_blur, mu2, cv::Size(11, 11), 1.5);

    // Squares and product of means.
    cv::Mat mu1_sq = mu1.mul(mu1);
    cv::Mat mu2_sq = mu2.mul(mu2);
    cv::Mat mu1_mu2 = mu1.mul(mu2);

    // Compute local variances and covariance.
    cv::Mat sigma1_sq, sigma2_sq, sigma12;
    cv::GaussianBlur(I.mul(I), sigma1_sq, cv::Size(11, 11), 1.5);
    sigma1_sq -= mu1_sq;
    cv::GaussianBlur(I_blur.mul(I_blur), sigma2_sq, cv::Size(11, 11), 1.5);
    sigma2_sq -= mu2_sq;
    cv::GaussianBlur(I.mul(I_blur), sigma12, cv::Size(11, 11), 1.5);
    sigma12 -= mu1_mu2;

    // Define constants for stability (using L = 255 for 8-bit images).
    double C1 = (0.01 * 255) * (0.01 * 255); // ~6.5025
    double C2 = (0.03 * 255) * (0.03 * 255); // ~58.5225

    // Compute the SSIM map.
    cv::Mat numerator = (2 * mu1_mu2 + C1).mul(2 * sigma12 + C2);
    cv::Mat denominator = (mu1_sq + mu2_sq + C1).mul(sigma1_sq + sigma2_sq + C2);
    cv::Mat ssim_map;
    cv::divide(numerator, denominator, ssim_map);

    // Build a histogram of the SSIM map using 256 bins.
    const int histSize = 256;
    std::vector<float> hist(histSize, 0.0f);

    // Iterate over each pixel in the SSIM map
    // (clamp the values within [0, 1] before binning)
    for (int i = 0; i < ssim_map.rows; i++)
    {
        for (int j = 0; j < ssim_map.cols; j++)
        {
            float val = ssim_map.at<float>(i, j);
            // Clamp the value to [0, 1]
            val = std::max(0.0f, std::min(1.0f, val));
            int bin = static_cast<int>(val * (histSize - 1));
            hist[bin] += 1.0f;
        }
    }

    // Normalize the histogram by the total number of pixels.
    float totalPixels = static_cast<float>(ssim_map.rows * ssim_map.cols);
    for (int i = 0; i < histSize; i++)
    {
        hist[i] /= totalPixels;
    }

    return hist;
}