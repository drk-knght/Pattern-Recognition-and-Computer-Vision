#include "ImageProcessor.h"
#include "Thresholder.h"
#include "KMeans.h"
#include "filter.h"
#include "morphology.h"
#include "RegionAnalyzer.h"
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

     // First apply closing with larger kernel to fill holes
    threshold_frame = Morphology::closing(threshold_frame, 7);  // Increased kernel size
    threshold_frame = Morphology::closing(threshold_frame, 7);  // Second pass
    
    // Then apply opening to remove any noise
    threshold_frame = Morphology::opening(threshold_frame, 3);

    // After getting the threshold frame, analyze regions
    threshold_frame = regionAnalyzer.analyzeAndVisualize(threshold_frame);

    imshow(WINDOW_ORIGINAL, frame);
    imshow(WINDOW_PROCESSED, threshold_frame);
    // imshow("Regions", regions_frame);
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