/*
    Agnibha Chatterjee
    Om Agarwal
    Jan 12 2024
    CS5330- Pattern Recognition & Computer Vision
    This file is the entry
*/

#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <sys/time.h>
#include <iostream>
#include <cmath>
#include "filter.h"

// Returns the current time in seconds
double getTime()
{
    struct timeval cur;
    gettimeofday(&cur, NULL);                      // Get current time
    return (cur.tv_sec + cur.tv_usec / 1000000.0); // Return time in seconds
}

// Customized grayscale filter by averaging the channels
int greyscale(cv::Mat &src, cv::Mat &dst)
{
    // Check if the source image is valid
    if (src.empty() || src.type() != CV_8UC3)
    {
        return -1; // Return error if invalid
    }

    // Split the source image into its color channels
    cv::Mat single_channel[3];
    cv::split(src, single_channel);

    // Calculate the average channel value
    cv::Mat avg_channel_value = (single_channel[0] + single_channel[1] + single_channel[2]) / 3;

    // Set all channels to the average value to create a grayscale image
    single_channel[0] = avg_channel_value;
    single_channel[1] = avg_channel_value;
    single_channel[2] = avg_channel_value;

    // Merge the channels back into the destination image
    cv::merge(single_channel, 3, dst);
    return 0; // Return success
}

// Sepia Filter
int sepia(cv::Mat &src, cv::Mat &dst)
{
    // Check if the source image is valid
    if (src.empty() || src.type() != CV_8UC3)
    {
        return -1; // Return error if invalid
    }

    // Initialize the destination image
    dst = cv::Mat::zeros(src.size(), CV_8UC3);

    // Apply sepia filter to each pixel
    for (int i = 0; i < src.rows; i++)
    {
        for (int j = 0; j < src.cols; j++)
        {
            // Get the pixel values from the source image
            int src_blue_pixel = src.at<cv::Vec3b>(i, j)[0];
            int src_green_pixel = src.at<cv::Vec3b>(i, j)[1];
            int src_red_pixel = src.at<cv::Vec3b>(i, j)[2];

            // Calculate the new pixel values using sepia coefficients
            float blue_dst = 0.131f * src_blue_pixel + 0.534f * src_green_pixel + 0.272f * src_red_pixel;
            float green_dst = (0.168f * src_blue_pixel + 0.686f * src_green_pixel + 0.349f * src_red_pixel) / 1.2;
            float red_dst = (0.189f * src_blue_pixel + 0.769f * src_green_pixel + 0.393f * src_red_pixel) / 1.35;

            // Set the new pixel values in the destination image
            dst.at<cv::Vec3b>(i, j)[0] = (unsigned char)(blue_dst);
            dst.at<cv::Vec3b>(i, j)[1] = (unsigned char)(green_dst);
            dst.at<cv::Vec3b>(i, j)[2] = (unsigned char)(red_dst);
        }
    }
    return 0; // Return success
}

// Gaussian Filter
int blur5x5_1(cv::Mat &src, cv::Mat &dst)
{
    // Check if the source image is valid
    if (src.empty() || src.type() != CV_8UC3)
    {
        return -1; // Return error if invalid
    }

    // Start timing the operation
    double startTime = getTime();
    src.copyTo(dst); // Copy source to destination
    int kernel[5][5] = {
        {1, 2, 4, 2, 1},
        {2, 4, 8, 4, 2},
        {4, 8, 16, 8, 4},
        {2, 4, 8, 4, 2},
        {1, 2, 4, 2, 1}};

    int normalize_value = 100; // Normalization value for the kernel

    // Apply the Gaussian blur using the kernel
    for (int i = 2; i < src.rows - 2; i++)
    {
        for (int j = 2; j < src.cols - 2; j++)
        {
            int blue = 0, green = 0, red = 0;

            // Convolve the kernel with the image
            for (int dx = -2; dx <= 2; dx++)
            {
                for (int dy = -2; dy <= 2; dy++)
                {
                    cv::Vec3b pixel = src.at<cv::Vec3b>(i + dx, j + dy);
                    int kernel_weight = kernel[dx + 2][dy + 2];

                    // Accumulate the weighted pixel values
                    blue += kernel_weight * pixel[0];
                    green += kernel_weight * pixel[1];
                    red += kernel_weight * pixel[2];
                }
            }
            // Set the blurred pixel value in the destination image
            dst.at<cv::Vec3b>(i, j)[0] = blue / normalize_value;
            dst.at<cv::Vec3b>(i, j)[1] = green / normalize_value;
            dst.at<cv::Vec3b>(i, j)[2] = red / normalize_value;
        }
    }

    // End timing and print the results
    double endTime = getTime();
    double difference = (endTime - startTime);
    printf("Time per image (1): %.4lf seconds\n", difference);
    return 0; // Return success
}

// 5x5 Gaussian Blur (alternative implementation)
int blur5x5_2(cv::Mat &src, cv::Mat &dst)
{
    // Check if the source image is valid
    if (src.empty() || src.type() != CV_8UC3)
    {
        return -1; // Return error if invalid
    }

    cv::Mat temp = cv::Mat::zeros(src.size(), CV_16SC3); // Temporary matrix for intermediate results
    src.copyTo(dst);                                     // Copy source to destination

    // Apply the Gaussian blur using a different method
    for (int i = 2; i + 2 < src.rows; i++)
    {
        for (int j = 2; j + 2 < src.cols; j++)
        {
            for (int channel = 0; channel < 3; channel++)
            {
                temp.at<cv::Vec3s>(i, j)[channel] = 1 * src.at<cv::Vec3b>(i, j - 2)[channel] +
                                                    2 * src.at<cv::Vec3b>(i, j - 1)[channel] +
                                                    1 * src.at<cv::Vec3b>(i, j + 2)[channel] +
                                                    2 * src.at<cv::Vec3b>(i, j + 1)[channel] +
                                                    4 * src.at<cv::Vec3b>(i, j)[channel];
            }
        }
    }

    // Normalize the blurred values and set them in the destination image
    for (int i = 2; i + 2 < src.rows; i++)
    {
        for (int j = 2; j + 2 < src.cols; j++)
        {
            for (int channel = 0; channel < 3; channel++)
            {
                int blur_value = 1 * temp.at<cv::Vec3s>(i - 2, j)[channel] +
                                 2 * temp.at<cv::Vec3s>(i - 1, j)[channel] +
                                 1 * temp.at<cv::Vec3s>(i + 2, j)[channel] +
                                 2 * temp.at<cv::Vec3s>(i + 1, j)[channel] +
                                 4 * temp.at<cv::Vec3s>(i, j)[channel];
                blur_value = blur_value / 100;                 // Normalize the value
                dst.at<cv::Vec3b>(i, j)[channel] = blur_value; // Set the blurred value
            }
        }
    }
    return 0; // Return success
}

// Positive Right Sobel X Filter
int sobelX3x3(cv::Mat &src, cv::Mat &dst)
{
    // Check if the source image is valid
    if (src.empty() || src.type() != CV_8UC3)
    {
        return -1; // Return error if invalid
    }

    dst = cv::Mat::zeros(src.size(), CV_16SC3);          // Initialize destination image
    cv::Mat temp = cv::Mat::zeros(src.size(), CV_16SC3); // Temporary matrix for intermediate results

    // Apply Sobel filter in the X direction
    for (int i = 1; i + 1 < src.rows; i++)
    {
        for (int j = 1; j + 1 < src.cols; j++)
        {
            for (int channel = 0; channel < 3; channel++)
            {
                temp.at<cv::Vec3s>(i, j)[channel] = -1 * src.at<cv::Vec3b>(i, j - 1)[channel] +
                                                    0 * src.at<cv::Vec3b>(i, j)[channel] +
                                                    1 * src.at<cv::Vec3b>(i, j + 1)[channel];
            }
        }
    }

    // Combine results to get the final Sobel X values
    int mx = 0; // Variable to track maximum value
    for (int i = 1; i + 1 < src.rows; i++)
    {
        for (int j = 1; j + 1 < src.cols; j++)
        {
            for (int channel = 0; channel < 3; channel++)
            {
                int temp_sobel_x_value = 1 * temp.at<cv::Vec3s>(i - 1, j)[channel] +
                                         2 * temp.at<cv::Vec3s>(i, j)[channel] +
                                         1 * temp.at<cv::Vec3s>(i + 1, j)[channel];
                temp_sobel_x_value = temp_sobel_x_value / 4;           // Normalize the value
                dst.at<cv::Vec3s>(i, j)[channel] = temp_sobel_x_value; // Set the Sobel X value
                if (temp_sobel_x_value > mx)
                {
                    mx = temp_sobel_x_value; // Update maximum value
                }
            }
        }
    }
    return 0; // Return success
}

// Positive UP Sobel Y Filter
int sobelY3x3(cv::Mat &src, cv::Mat &dst)
{
    // Check if the source image is valid
    if (src.empty() || src.type() != CV_8UC3)
    {
        return -1; // Return error if invalid
    }

    dst = cv::Mat::zeros(src.size(), CV_16SC3);          // Initialize destination image
    cv::Mat temp = cv::Mat::zeros(src.size(), CV_16SC3); // Temporary matrix for intermediate results

    // Apply Sobel filter in the Y direction
    for (int i = 1; i + 1 < src.rows; i++)
    {
        for (int j = 1; j + 1 < src.cols; j++)
        {
            for (int channel = 0; channel < 3; channel++)
            {
                temp.at<cv::Vec3s>(i, j)[channel] = -1 * src.at<cv::Vec3b>(i - 1, j)[channel] +
                                                    0 * src.at<cv::Vec3b>(i, j)[channel] +
                                                    1 * src.at<cv::Vec3b>(i + 1, j)[channel];
            }
        }
    }

    // Combine results to get the final Sobel Y values
    for (int i = 1; i + 1 < src.rows; i++)
    {
        for (int j = 1; j + 1 < src.cols; j++)
        {
            for (int channel = 0; channel < 3; channel++)
            {
                int temp_sobel_y_value = 1 * temp.at<cv::Vec3s>(i, j - 1)[channel] +
                                         2 * temp.at<cv::Vec3s>(i, j)[channel] +
                                         1 * temp.at<cv::Vec3s>(i, j + 1)[channel];
                temp_sobel_y_value = temp_sobel_y_value / 4;           // Normalize the value
                dst.at<cv::Vec3s>(i, j)[channel] = temp_sobel_y_value; // Set the Sobel Y value
            }
        }
    }
    return 0; // Return success
}

// Calculate the magnitude of the gradient
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst)
{
    // Check if the input images are valid
    if (sx.empty() || sy.empty())
    {
        return -1; // Return error if invalid
    }
    if (sx.type() != CV_16SC3 || sy.type() != CV_16SC3)
    {
        return -1; // Return error if types are incorrect
    }

    dst = cv::Mat::zeros(sx.size(), CV_8UC3); // Initialize destination image

    // Calculate the magnitude of the gradient
    for (int i = 0; i < sx.rows; i++)
    {
        for (int j = 0; j < sx.cols; j++)
        {
            for (int channel = 0; channel < 3; channel++)
            {
                float sx_val = sx.at<cv::Vec3s>(i, j)[channel];
                float sy_val = sy.at<cv::Vec3s>(i, j)[channel];
                float temp = sqrt(sx_val * sx_val + sy_val * sy_val); // Calculate magnitude
                dst.at<cv::Vec3b>(i, j)[channel] = temp / 1.4f;       // Normalize and set in destination
            }
        }
    }
    return 0; // Return success
}

// Quantize the image colors after blurring
int blurQuantize(cv::Mat &src, cv::Mat &dst, int levels)
{
    // Check if the source image is valid
    if (src.empty() || src.type() != CV_8UC3)
    {
        return -1; // Return error if invalid
    }

    cv::Mat temp = cv::Mat::zeros(src.size(), CV_8UC3); // Temporary matrix for blurred image
    int res = blur5x5_2(src, temp);                     // Apply blur

    if (res == -1)
    {
        return -1; // Return error if blur fails
    }

            for(int channel=0;channel<3;channel++){
                int xt=temp.at<cv::Vec3b>(i,j)[channel];
                xt=xt/b;
                int xf=xt*b;
                dst.at<cv::Vec3b>(i,j)[channel]=xf;
            }
        }
    }
    return 0; // Return success
}

// Isolate red colors in the image
int isolateRed(cv::Mat &src, cv::Mat dst)
{
    // Check if the source image is valid
    if (src.empty() || src.type() != CV_8UC3)
    {
        return -1; // Return error if invalid
    }

    cv::Mat hsv, mask, grey;                                                      // Matrices for HSV, mask, and grey image
    cv::cvtColor(dst, hsv, cv::COLOR_BGR2HSV);                                    // Convert to HSV color space
    cv::inRange(hsv, cv::Scalar(160, 100, 100), cv::Scalar(180, 255, 255), mask); // Create mask for red colors

    cv::cvtColor(dst, grey, cv::COLOR_BGR2GRAY); // Convert to grayscale
    cv::cvtColor(grey, dst, cv::COLOR_GRAY2BGR); // Convert back to BGR

    cv::Mat colored;            // Matrix for colored output
    src.copyTo(colored, mask);  // Copy only the red regions to the colored image
    cv::add(dst, colored, dst); // Add the colored regions to the grey image

    return 0; // Return success
}
