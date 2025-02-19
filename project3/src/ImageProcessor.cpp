#include "ImageProcessor.h"
#include "Thresholder.h"
#include "KMeans.h"
#include "filter.h"
#include "morphology.h"
#include "RegionAnalyzer.h"

#include <fstream>
#include <iostream>

using namespace cv;
using namespace std;

ImageProcessor::ImageProcessor() : gen(rd()) {}

Mat ImageProcessor::preprocess_image(const Mat &frame)
{
    Mat hsv, blurred, gray, darkened;

    cvtColor(frame, hsv, COLOR_BGR2HSV);

    vector<Mat> hsv_channels;
    split(hsv, hsv_channels);
    Mat saturation = hsv_channels[1];

    // GaussianBlur(frame, blurred, Size(5, 5), 0);
    gaussian_blur(frame, blurred);
    cvtColor(blurred, gray, COLOR_BGR2GRAY);
    darkened = gray.clone();

    for (int i = 0; i < darkened.rows; i++)
    {
        for (int j = 0; j < darkened.cols; j++)
        {
            if (saturation.at<uchar>(i, j) > 50)
            {
                darkened.at<uchar>(i, j) = static_cast<uchar>(darkened.at<uchar>(i, j) * 0.7);
            }
        }
    }

    return darkened;
}

int ImageProcessor::get_dynamic_threshold(const Mat &gray_image)
{
    vector<float> pixels;
    int sample_size = (gray_image.rows * gray_image.cols) / 16;

    uniform_int_distribution<> dis_rows(0, gray_image.rows - 1);
    uniform_int_distribution<> dis_cols(0, gray_image.cols - 1);

    for (int i = 0; i < sample_size; i++)
    {
        int row = dis_rows(gen);
        int col = dis_cols(gen);
        pixels.push_back(static_cast<float>(gray_image.at<uchar>(row, col)));
    }

    // Convert pixels to a Mat for KMeans
    cv::Mat samples(pixels.size(), 1, CV_32F);
    for (size_t i = 0; i < pixels.size(); i++)
    {
        samples.at<float>(i) = pixels[i];
    }

    // Use KMeans class
    KMeans kmeans(2); // 2 clusters
    kmeans.fit(samples);

    const cv::Mat &kmeans_centers = kmeans.get_centroids();
    return static_cast<int>((kmeans_centers.at<float>(0) + kmeans_centers.at<float>(1)) / 2);
}

void ImageProcessor::process_frame(const Mat &frame, Mat &threshold_frame)
{
    Mat processed = preprocess_image(frame);
    int threshold_value = get_dynamic_threshold(processed);

    // Use our custom Thresholder instead of cv::threshold
    threshold_frame = Thresholder::apply(processed, threshold_value);

    // Apply morphological filtering
    threshold_frame = Morphology::closing(threshold_frame, 7); // Increased kernel size
    threshold_frame = Morphology::closing(threshold_frame, 7); // Second pass
    threshold_frame = Morphology::opening(threshold_frame, 3);

    // Save the single-channel binary image for training data extraction
    // (This image is 1-channel and is suitable for connectedComponentsWithStats)
    regions_frame = threshold_frame.clone();

    // Generate and display a colored visualization using region analysis
    Mat visualization = regionAnalyzer.analyzeAndVisualize(threshold_frame);
    threshold_frame = visualization;

    imshow(WINDOW_ORIGINAL, frame);
    imshow(WINDOW_PROCESSED, threshold_frame);
}

void ImageProcessor::create_windows()
{
    namedWindow(WINDOW_ORIGINAL, WINDOW_AUTOSIZE);
    namedWindow(WINDOW_PROCESSED, WINDOW_AUTOSIZE);
}

void ImageProcessor::destroy_windows()
{
    destroyAllWindows();
}

Mat ImageProcessor::get_threshold_frame()
{
    return regions_frame;
}

void ImageProcessor::collect_training_data(const Mat &binaryFrame)
{
    // Ensure the input image is single-channel.
    Mat gray;
    if (binaryFrame.channels() > 1)
    {
        cvtColor(binaryFrame, gray, COLOR_BGR2GRAY);
    }
    else
    {
        gray = binaryFrame;
    }

    Mat labels, stats, centroids;
    int numLabels = connectedComponentsWithStats(gray, labels, stats, centroids, 8, CV_32S);
    int bestLabel = -1;
    int bestArea = 0;
    for (int i = 1; i < numLabels; i++)
    {
        int area = stats.at<int>(i, CC_STAT_AREA);
        if (area > bestArea)
        {
            bestArea = area;
            bestLabel = i;
        }
    }

    if (bestLabel == -1)
    {
        cerr << "No valid object found for training data." << endl;
        return;
    }

    int left = stats.at<int>(bestLabel, CC_STAT_LEFT);
    int top = stats.at<int>(bestLabel, CC_STAT_TOP);
    int width = stats.at<int>(bestLabel, CC_STAT_WIDTH);
    int height = stats.at<int>(bestLabel, CC_STAT_HEIGHT);
    double boundingBoxArea = static_cast<double>(width * height);
    double percentFilled = bestArea / boundingBoxArea;
    double heightWidthRatio = static_cast<double>(height) / width;

    Mat mask = (labels == bestLabel);
    Moments m = moments(mask, true);
    double centroidX = (m.m00 != 0) ? (m.m10 / m.m00) : 0;
    double centroidY = (m.m00 != 0) ? (m.m01 / m.m00) : 0;

    // Prompt the user for the object's label
    string objectLabel;
    cout << "Enter label for the detected object: ";
    cin >> objectLabel;

    // Append the feature vector (and label) to a CSV file
    ofstream outFile("training_data.csv", ios::app);
    if (!outFile.is_open())
    {
        cerr << "Error opening training_data.csv for writing." << endl;
        return;
    }
    // CSV columns: label, area, percent_filled, height_width_ratio, centroid_x, centroid_y, left, top, width, height
    outFile << objectLabel << "," << bestArea << "," << percentFilled << "," << heightWidthRatio << ","
            << centroidX << "," << centroidY << "," << left << "," << top << "," << width << "," << height << "\n";
    outFile.close();

    cout << "Training data saved for object: " << objectLabel << endl;
}