#include "morphology.h"

cv::Mat Morphology::erode(const cv::Mat &input, int kernel_size)
{
    cv::Mat output = cv::Mat::zeros(input.size(), input.type());
    int offset = kernel_size / 2;

    for (int i = offset; i < input.rows - offset; i++)
    {
        for (int j = offset; j < input.cols - offset; j++)
        {
            output.at<uchar>(i, j) = checkNeighborhood(input, i, j, kernel_size, true) ? 255 : 0;
        }
    }
    return output;
}

cv::Mat Morphology::dilate(const cv::Mat &input, int kernel_size)
{
    cv::Mat output = cv::Mat::zeros(input.size(), input.type());
    int offset = kernel_size / 2;

    for (int i = offset; i < input.rows - offset; i++)
    {
        for (int j = offset; j < input.cols - offset; j++)
        {
            output.at<uchar>(i, j) = checkNeighborhood(input, i, j, kernel_size, false) ? 255 : 0;
        }
    }
    return output;
}

cv::Mat Morphology::opening(const cv::Mat &input, int kernel_size)
{
    cv::Mat eroded = erode(input, kernel_size);
    return dilate(eroded, kernel_size);
}

cv::Mat Morphology::closing(const cv::Mat &input, int kernel_size)
{
    // More aggressive dilation
    cv::Mat dilated = dilate(input, kernel_size);
    dilated = dilate(dilated, kernel_size); // Second dilation pass

    // Single erosion pass to maintain shape
    return erode(dilated, kernel_size);
}

bool Morphology::checkNeighborhood(const cv::Mat &input, int row, int col, int kernel_size, bool isErosion)
{
    int offset = kernel_size / 2;
    int whiteCount = 0;
    int totalPixels = 0;

    for (int i = -offset; i <= offset; i++)
    {
        for (int j = -offset; j <= offset; j++)
        {
            // Skip corners of the square kernel to make it more circular
            if (abs(i) + abs(j) > kernel_size)
                continue;

            bool isWhite = input.at<uchar>(row + i, col + j) == 255;
            whiteCount += isWhite ? 1 : 0;
            totalPixels++;
        }
    }

    if (isErosion)
    {
        // For erosion, require at least 80% of pixels to be white
        return (float)whiteCount / totalPixels > 0.8;
    }
    else
    {
        // For dilation, require at least 20% of pixels to be white
        return (float)whiteCount / totalPixels > 0.2;
    }
}