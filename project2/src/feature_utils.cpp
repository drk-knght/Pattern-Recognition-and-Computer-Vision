#include "feature_utils.h"

#include <opencv2/dnn.hpp> // New: for DNN support
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <cmath>
#include <algorithm>
#include <limits>
#include <iostream>

std::vector<float> extractCenterPatch(const cv::Mat &img)
{
    std::vector<float> features;

    int rows = img.rows;
    int cols = img.cols;

    int center_row = rows / 2;
    int center_col = cols / 2;

    int start_row = center_row - 3;
    int end_row = center_row + 3;
    int start_col = center_col - 3;
    int end_col = center_col + 3;

    for (int i = start_row; i <= end_row; i++)
    {
        for (int j = start_col; j <= end_col; j++)
        {
            cv::Vec3b pixel = img.at<cv::Vec3b>(i, j);

            features.push_back((float)pixel[0]); // Blue
            features.push_back((float)pixel[1]); // Green
            features.push_back((float)pixel[2]); // Red
        }
    }
    return features;
}

// Calculate Sum of Squared Differences
float calculateSSD(const std::vector<float> &feat1, const std::vector<float> &feat2)
{
    float ssd = 0;
    for (size_t i = 0; i < feat1.size(); i++)
    {
        float diff = feat1[i] - feat2[i];
        ssd += diff * diff;
    }
    return ssd;
}

std::vector<float> extractColorHistogram(const cv::Mat &img)
{
    std::vector<float> features;
    const int bins = 16; // 16 bins for each channel
    const int total_bins = bins * bins;

    // Initialize histogram with zeros
    std::vector<float> hist(total_bins, 0.0f);
    float total_pixels = 0.0f;

    // Convert to float and split channels
    cv::Mat float_img;
    img.convertTo(float_img, CV_32F, 1.0 / 255.0);
    std::vector<cv::Mat> channels;
    cv::split(float_img, channels);

    // Calculate r and g chromaticity for each pixel
    for (int y = 0; y < img.rows; y++)
    {
        for (int x = 0; x < img.cols; x++)
        {
            float b = channels[0].at<float>(y, x);
            float g = channels[1].at<float>(y, x);
            float r = channels[2].at<float>(y, x);
            float sum = r + g + b;

            if (sum > 0.0f)
            {
                // Calculate r and g chromaticity
                float r_chrom = r / sum;
                float g_chrom = g / sum;

                // Calculate bin indices
                int r_bin = std::min((int)(r_chrom * bins), bins - 1);
                int g_bin = std::min((int)(g_chrom * bins), bins - 1);

                // Update histogram
                hist[r_bin * bins + g_bin] += 1.0f;
                total_pixels += 1.0f;
            }
        }
    }

    // Normalize histogram
    if (total_pixels > 0)
    {
        for (float &bin : hist)
        {
            bin /= total_pixels;
        }
    }

    return hist;
}

// Calculate histogram intersection (higher value means more similar)
float calculateHistIntersection(const std::vector<float> &hist1, const std::vector<float> &hist2)
{
    if (hist1.size() != hist2.size())
    {
        printf("Error: Histograms have different sizes\n");
        return 0.0f;
    }

    float intersection = 0.0f;
    for (size_t i = 0; i < hist1.size(); i++)
    {
        intersection += std::min(hist1[i], hist2[i]);
    }

    return intersection;
}

// Spatial histogram implementation
std::vector<float> extractSpatialColorHistogram(const cv::Mat &img)
{
    // Parameters for histogram
    int histSize = 8;
    float range[] = {0, 256};
    const float *histRange = {range};
    std::vector<float> combined_features;

    // Split image into top and bottom halves
    int height = img.rows;
    cv::Mat top_half = img(cv::Range(0, height / 2), cv::Range::all());
    cv::Mat bottom_half = img(cv::Range(height / 2, height), cv::Range::all());

    // Process each half
    std::vector<cv::Mat> halves = {top_half, bottom_half};

    for (const cv::Mat &half : halves)
    {
        // Direct access to pixels using BGR order
        std::vector<std::vector<float>> channel_hists(3, std::vector<float>(histSize, 0.0f));
        float total_pixels = half.rows * half.cols;

        // Count pixels for each channel directly
        for (int y = 0; y < half.rows; y++)
        {
            for (int x = 0; x < half.cols; x++)
            {
                cv::Vec3b pixel = half.at<cv::Vec3b>(y, x);

                // Calculate bin for each channel (B,G,R)
                for (int c = 0; c < 3; c++)
                {
                    int bin = (pixel[c] * histSize) / 256;
                    channel_hists[c][bin]++;
                }
            }
        }

        // Normalize and add all channels to feature vector
        for (auto &hist : channel_hists)
        {
            for (float bin_count : hist)
            {
                combined_features.push_back(bin_count / total_pixels);
            }
        }
    }

    return combined_features;
}

float calculateSpatialHistIntersection(const std::vector<float> &hist1, const std::vector<float> &hist2)
{
    float similarity = 0;
    int bins_per_half = 24; // 8 bins * 3 channels

    // Calculate intersection for top half
    float top_similarity = 0;
    for (int i = 0; i < bins_per_half; i++)
    {
        top_similarity += std::min(hist1[i], hist2[i]);
    }

    // Calculate intersection for bottom half
    float bottom_similarity = 0;
    for (int i = bins_per_half; i < bins_per_half * 2; i++)
    {
        bottom_similarity += std::min(hist1[i], hist2[i]);
    }

    // Equal weighting of top and bottom similarities
    similarity = (top_similarity + bottom_similarity) / 2.0;

    return similarity;
}

// Positive Right Sobel X Filter
/*
    -1 0 1
    0 0 0
    -2 0 2
*/
int sobelX3x3(cv::Mat &src, cv::Mat &dst)
{
    if (src.empty() || src.type() != CV_8UC3)
    {
        return -1;
    }
    dst = cv::Mat::zeros(src.size(), CV_16SC3);
    cv::Mat temp = cv::Mat::zeros(src.size(), CV_16SC3);
    for (int i = 1; i + 1 < src.rows; i++)
    {
        for (int j = 1; j + 1 < src.cols; j++)
        {
            for (int channel = 0; channel < 3; channel++)
            {
                temp.at<cv::Vec3s>(i, j)[channel] = -1 * src.at<cv::Vec3b>(i, j - 1)[channel] + 1 * src.at<cv::Vec3b>(i, j + 1)[channel];
            }
        }
    }
    for (int i = 1; i + 1 < src.rows; i++)
    {
        for (int j = 1; j + 1 < src.cols; j++)
        {
            for (int channel = 0; channel < 3; channel++)
            {
                int temp_sobel_x_value = 1 * temp.at<cv::Vec3s>(i - 1, j)[channel] + 2 * temp.at<cv::Vec3s>(i, j)[channel] + 1 * temp.at<cv::Vec3s>(i + 1, j)[channel];
                temp_sobel_x_value = temp_sobel_x_value / 4;
                dst.at<cv::Vec3s>(i, j)[channel] = temp_sobel_x_value;
            }
        }
    }
    return 0;
}

// Positive UP Sobel Y Filter
/*
    1 2 1
    0 0 0
    -1 -2 -1
*/
int sobelY3x3(cv::Mat &src, cv::Mat &dst)
{
    if (src.empty() || src.type() != CV_8UC3)
    {
        return -1;
    }
    dst = cv::Mat::zeros(src.size(), CV_16SC3);
    cv::Mat temp = cv::Mat::zeros(src.size(), CV_16SC3);
    for (int i = 1; i + 1 < src.rows; i++)
    {
        for (int j = 1; j + 1 < src.cols; j++)
        {
            for (int channel = 0; channel < 3; channel++)
            {
                temp.at<cv::Vec3s>(i, j)[channel] = -1 * src.at<cv::Vec3b>(i - 1, j)[channel] + 1 * src.at<cv::Vec3b>(i + 1, j)[channel];
            }
        }
    }
    for (int i = 1; i + 1 < src.rows; i++)
    {
        for (int j = 1; j + 1 < src.cols; j++)
        {
            for (int channel = 0; channel < 3; channel++)
            {
                int16_t temp_sobel_x_value = 1 * temp.at<cv::Vec3s>(i, j - 1)[channel] + 2 * temp.at<cv::Vec3s>(i, j)[channel] + 1 * temp.at<cv::Vec3s>(i, j + 1)[channel];
                temp_sobel_x_value = temp_sobel_x_value / 4;
                dst.at<cv::Vec3s>(i, j)[channel] = temp_sobel_x_value;
            }
        }
    }
    return 0;
}

std::vector<float> extractTextureHistogram(const cv::Mat &img)
{
    // Create a copy of input image
    cv::Mat img_copy;
    img.copyTo(img_copy);

    // Calculate Sobel gradients using custom filters
    cv::Mat sobelX, sobelY;
    sobelX3x3(img_copy, sobelX);
    sobelY3x3(img_copy, sobelY);

    // Convert to CV_32F type
    cv::Mat sobelX_float, sobelY_float;
    sobelX.convertTo(sobelX_float, CV_32F);
    sobelY.convertTo(sobelY_float, CV_32F);

    // Calculate magnitude
    cv::Mat magnitude(img.size(), CV_32F);
    for (int i = 0; i < img.rows; i++)
    {
        for (int j = 0; j < img.cols; j++)
        {
            float sumSquares = 0.0f;
            for (int c = 0; c < 3; c++)
            {
                float dx = sobelX_float.at<cv::Vec3f>(i, j)[c];
                float dy = sobelY_float.at<cv::Vec3f>(i, j)[c];
                sumSquares += dx * dx + dy * dy;
            }
            magnitude.at<float>(i, j) = std::sqrt(sumSquares);
        }
    }

    // Calculate histogram of gradient magnitudes manually
    std::vector<float> hist(32, 0.0f); // 32 bins initialized to 0
    float bin_width = 512.0f / 32.0f;  // Width of each bin (512/32)
    float total_pixels = 0.0f;

    // Count values into bins
    for (int i = 0; i < magnitude.rows; i++)
    {
        for (int j = 0; j < magnitude.cols; j++)
        {
            float mag = magnitude.at<float>(i, j);
            int bin = std::min(static_cast<int>(mag / bin_width), 31); // Clamp to last bin if exceeds range
            hist[bin] += 1.0f;
            total_pixels += 1.0f;
        }
    }

    // Normalize histogram (each bin value divided by total count)
    if (total_pixels > 0)
    {
        for (float &bin : hist)
        {
            bin /= total_pixels;
        }
    }

    return hist;
}

std::vector<float> extractCombinedFeatures(const cv::Mat &img)
{
    // Get color histogram
    std::vector<float> color_hist = extractColorHistogram(img);

    // Get texture histogram
    std::vector<float> texture_hist = extractTextureHistogram(img);

    // Combine both features
    std::vector<float> combined;
    combined.insert(combined.end(), color_hist.begin(), color_hist.end());
    combined.insert(combined.end(), texture_hist.begin(), texture_hist.end());

    return combined;
}

float calculateCombinedDistance(const std::vector<float> &feat1, const std::vector<float> &feat2)
{
    // Assuming first half is color histogram and second half is texture histogram
    int half_size = feat1.size() / 2;

    // Calculate color histogram intersection
    float color_sim = 0;
    for (int i = 0; i < half_size; i++)
    {
        color_sim += std::min(feat1[i], feat2[i]);
    }

    // Calculate texture histogram intersection
    float texture_sim = 0;
    for (int i = half_size; i < feat1.size(); i++)
    {
        texture_sim += std::min(feat1[i], feat2[i]);
    }

    // Weight both equally and return combined similarity
    return (color_sim + texture_sim) / 2.0f;
}

std::vector<float> extractColorSpatialVariance(const cv::Mat &img)
{
    // Convert the image to the HSV colorspace.
    cv::Mat hsv;
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);

    // Threshold for a yellow color range typical of bananas.
    // (Adjust these values if your bananas vary in color.)
    cv::Scalar lower_yellow(15, 100, 100);
    cv::Scalar upper_yellow(35, 255, 255);
    cv::Mat mask;
    cv::inRange(hsv, lower_yellow, upper_yellow, mask);

    // Find contours in the mask.
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // If a yellow region was detected, pick the largest one as the region of interest.
    cv::Rect roi;
    if (!contours.empty())
    {
        double max_area = 0.0;
        int max_idx = -1;
        for (size_t i = 0; i < contours.size(); i++)
        {
            double area = cv::contourArea(contours[i]);
            if (area > max_area)
            {
                max_area = area;
                max_idx = static_cast<int>(i);
            }
        }
        if (max_idx >= 0)
            roi = cv::boundingRect(contours[max_idx]);
    }

    // If a valid ROI was found (and not trivially small), crop to that region.
    cv::Mat candidate;
    if (roi.area() > 0 && roi.width > 10 && roi.height > 10)
    {
        candidate = img(roi);
    }
    else
    {
        // Fall back to the entire image if no valid yellow region is detected.
        candidate = img;
    }

    // Resize to a fixed size so that grid cells are comparable across images.
    cv::Mat resized;
    cv::resize(candidate, resized, cv::Size(256, 256));

    // Convert to a float image in the range [0, 1] and split into channels.
    cv::Mat float_img;
    resized.convertTo(float_img, CV_32F, 1.0 / 255.0);
    std::vector<cv::Mat> channels;
    cv::split(float_img, channels);

    // Create a grid (4x4) for each channel.
    const int grid_size = 4;
    int cell_height = resized.rows / grid_size;
    int cell_width = resized.cols / grid_size;
    std::vector<float> features;

    // For each color channel (B, G, R) in order.
    for (int c = 0; c < 3; c++)
    {
        // For each grid cell
        for (int i = 0; i < grid_size; i++)
        {
            for (int j = 0; j < grid_size; j++)
            {
                int x = j * cell_width;
                int y = i * cell_height;
                cv::Rect cell_rect(x, y, cell_width, cell_height);
                cv::Mat cell = channels[c](cell_rect);

                cv::Scalar mean, stddev;
                cv::meanStdDev(cell, mean, stddev);

                // Append the mean and the variance (stddev^2) for this cell.
                features.push_back(static_cast<float>(mean[0]));
                features.push_back(static_cast<float>(stddev[0] * stddev[0]));
            }
        }
    }

    return features;
}

float calculateSpatialVarianceDistance(const std::vector<float> &feat1, const std::vector<float> &feat2)
{
    if (feat1.size() != feat2.size())
    {
        return std::numeric_limits<float>::max();
    }

    float distance = 0.0f;
    // Weight for differences in variance (tweak this if needed).
    const float variance_weight = 1.5f;

    // Each grid cell is represented by two consecutive features: (mean, variance)
    for (size_t i = 0; i < feat1.size(); i += 2)
    {
        float mean_diff = feat1[i] - feat2[i];
        float var_diff = feat1[i + 1] - feat2[i + 1];

        // Sum of squared differences: treat variance differences with extra weight.
        distance += (mean_diff * mean_diff) + (variance_weight * var_diff * var_diff);
    }

    return distance;
}

// -----------------------------------------------------------------------------
// New DNN and combined feature functions
// -----------------------------------------------------------------------------

// Function to extract features using a deep neural network.
// Requires a pretrained model (e.g., an ONNX model). Adjust model path and preprocessing as needed.
std::vector<float> extractDnnFeatures(const cv::Mat &img)
{
    // Load the pretrained network only once.
    static cv::dnn::Net net = cv::dnn::readNetFromONNX("../resnet18-v2-7.onnx");
    if (net.empty())
    {
        std::cerr << "Error: Unable to load the DNN model." << std::endl;
        return std::vector<float>();
    }

    // Preprocess the image: resize to 224x224 and normalize.
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(224, 224));
    cv::Mat blob = cv::dnn::blobFromImage(resized, 1.0 / 255.0, cv::Size(224, 224), cv::Scalar(0, 0, 0), true, false);

    net.setInput(blob);
    cv::Mat output = net.forward();

    std::vector<float> dnn_features;
    dnn_features.assign((float *)output.datastart, (float *)output.dataend);
    return dnn_features;
}

// Function to extract a combined feature vector from an image by fusing DNN and spatial variance features.
std::vector<float> extractCombinedDnnSpatialVarianceFeatures(const cv::Mat &img)
{
    // Extract DNN features.
    std::vector<float> dnn_features = extractDnnFeatures(img);
    // Extract spatial variance features.
    std::vector<float> spatial_features = extractColorSpatialVariance(img);

    // Fuse the features by simple concatenation.
    std::vector<float> combined_features;
    combined_features.reserve(dnn_features.size() + spatial_features.size());
    combined_features.insert(combined_features.end(), dnn_features.begin(), dnn_features.end());
    combined_features.insert(combined_features.end(), spatial_features.begin(), spatial_features.end());

    return combined_features;
}