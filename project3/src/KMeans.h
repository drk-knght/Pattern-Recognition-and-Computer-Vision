#ifndef KMEANS_HPP
#define KMEANS_HPP

#include <opencv2/opencv.hpp>
#include <vector>

class KMeans {
public:
    KMeans(int k, int max_iterations = 100);
    void fit(const cv::Mat& data);
    const cv::Mat& get_centroids() const;
    const std::vector<int>& get_labels() const;

private:
    int k; // Number of clusters
    int max_iterations; // Maximum number of iterations
    cv::Mat centroids; // Centroids of the clusters
    std::vector<int> labels; // Labels for each data point

    void initialize_centroids(const cv::Mat& data);
    void assign_labels(const cv::Mat& data);
    void update_centroids(const cv::Mat& data);
};

#endif // KMEANS_HPP