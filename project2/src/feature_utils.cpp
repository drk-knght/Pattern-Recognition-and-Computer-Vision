#include "feature_utils.h"

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

std::vector<float> extractColorHistogram(const cv::Mat &img) {
    std::vector<float> features;
    const int bins = 16; // 16 bins for each channel
    const int total_bins = bins * bins;
    
    // Initialize histogram with zeros
    std::vector<float> hist(total_bins, 0.0f);
    float total_pixels = 0.0f;
    
    // Convert to float and split channels
    cv::Mat float_img;
    img.convertTo(float_img, CV_32F, 1.0/255.0);
    std::vector<cv::Mat> channels;
    cv::split(float_img, channels);
    
    // Calculate r and g chromaticity for each pixel
    for(int y = 0; y < img.rows; y++) {
        for(int x = 0; x < img.cols; x++) {
            float b = channels[0].at<float>(y, x);
            float g = channels[1].at<float>(y, x);
            float r = channels[2].at<float>(y, x);
            float sum = r + g + b;
            
            if(sum > 0.0f) {
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
    if(total_pixels > 0) {
        for(float& bin : hist) {
            bin /= total_pixels;
        }
    }
    
    return hist;
}

// Calculate histogram intersection (higher value means more similar)
float calculateHistIntersection(const std::vector<float> &hist1, const std::vector<float> &hist2) {
    if(hist1.size() != hist2.size()) {
        printf("Error: Histograms have different sizes\n");
        return 0.0f;
    }
    
    float intersection = 0.0f;
    for(size_t i = 0; i < hist1.size(); i++) {
        intersection += std::min(hist1[i], hist2[i]);
    }
    
    return intersection;
}

// Spatial histogram implementation
std::vector<float> extractSpatialColorHistogram(const cv::Mat &img) {

    // Parameters for histogram
    int histSize = 8;
    float range[] = {0, 256};
    const float* histRange = {range};
    std::vector<float> combined_features;

    // Split image into top and bottom halves
    int height = img.rows;
    cv::Mat top_half = img(cv::Range(0, height/2), cv::Range::all());
    cv::Mat bottom_half = img(cv::Range(height/2, height), cv::Range::all());

    // Process each half
    std::vector<cv::Mat> halves = {top_half, bottom_half};
    
    for (const cv::Mat& half : halves) {
        std::vector<cv::Mat> channels;
        cv::split(half, channels);
        
        // Calculate histogram for each channel
        for (int i = 0; i < 3; i++) {
            cv::Mat hist;
            cv::calcHist(&channels[i], 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);
            cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX);
            
            // Add to feature vector
            for (int j = 0; j < histSize; j++) {
                combined_features.push_back(hist.at<float>(j));
            }
        }
    }
    
    return combined_features;
}

float calculateSpatialHistIntersection(const std::vector<float>& hist1, const std::vector<float>& hist2) {
    float similarity = 0;
    int bins_per_half = 24; // 8 bins * 3 channels
    
    // Calculate intersection for top half
    float top_similarity = 0;
    for (int i = 0; i < bins_per_half; i++) {
        top_similarity += std::min(hist1[i], hist2[i]);
    }
    
    // Calculate intersection for bottom half
    float bottom_similarity = 0;
    for (int i = bins_per_half; i < bins_per_half * 2; i++) {
        bottom_similarity += std::min(hist1[i], hist2[i]);
    }
    
    // Equal weighting of top and bottom similarities
    similarity = (top_similarity + bottom_similarity) / 2.0;
    
    return similarity;
}

std::vector<float> extractTextureHistogram(const cv::Mat &img) {
    // Convert to grayscale
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    
    // Calculate Sobel gradients
    cv::Mat sobelX, sobelY;
    cv::Sobel(gray, sobelX, CV_32F, 1, 0);
    cv::Sobel(gray, sobelY, CV_32F, 0, 1);
    
    // Calculate magnitude
    cv::Mat magnitude;
    cv::magnitude(sobelX, sobelY, magnitude);
    
    // Calculate histogram of gradient magnitudes
    const int hist_size = 32;  // Number of bins
    float range[] = {0, 512};  // Range of magnitude values
    const float* hist_range = {range};
    
    cv::Mat hist;
    cv::calcHist(&magnitude, 1, 0, cv::Mat(), hist, 1, &hist_size, &hist_range);
    
    // Normalize histogram
    cv::normalize(hist, hist, 1, 0, cv::NORM_L1);
    
    // Convert to vector
    std::vector<float> features;
    hist.copyTo(features);
    return features;
}

std::vector<float> extractCombinedFeatures(const cv::Mat &img) {
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

float calculateCombinedDistance(const std::vector<float> &feat1, const std::vector<float> &feat2) {
    // Assuming first half is color histogram and second half is texture histogram
    int half_size = feat1.size() / 2;
    
    // Calculate color histogram intersection
    float color_sim = 0;
    for (int i = 0; i < half_size; i++) {
        color_sim += std::min(feat1[i], feat2[i]);
    }
    
    // Calculate texture histogram intersection
    float texture_sim = 0;
    for (int i = half_size; i < feat1.size(); i++) {
        texture_sim += std::min(feat1[i], feat2[i]);
    }
    
    // Weight both equally and return combined similarity
    return (color_sim + texture_sim) / 2.0f;
}