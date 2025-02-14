// project3/src/KMeans.cpp
#include "KMeans.h"
#include <cstdlib>
#include <limits>
#include <algorithm>
#include <random>

KMeans::KMeans(int k, int max_iterations) : k(k), max_iterations(max_iterations) {}

void KMeans::fit(const cv::Mat &data)
{
    initialize_centroids(data);
    labels = std::vector<int>(data.rows, 0);

    for (int iteration = 0; iteration < max_iterations; ++iteration)
    {
        assign_labels(data);
        update_centroids(data);
    }
}

const cv::Mat &KMeans::get_centroids() const
{
    return centroids;
}

const std::vector<int> &KMeans::get_labels() const
{
    return labels;
}

void KMeans::initialize_centroids(const cv::Mat &data)
{
    centroids = cv::Mat(k, data.cols, data.type());
    std::vector<int> indices(data.rows);
    for (int i = 0; i < data.rows; ++i)
    {
        indices[i] = i;
    }
    std::random_device rd;
    std::default_random_engine eng(rd());
    std::shuffle(indices.begin(), indices.end(), eng);
    for (int i = 0; i < k; ++i)
    {
        data.row(indices[i]).copyTo(centroids.row(i));
    }
}

void KMeans::assign_labels(const cv::Mat &data)
{
    for (int i = 0; i < data.rows; ++i)
    {
        double min_dist = std::numeric_limits<double>::max();
        for (int j = 0; j < k; ++j)
        {
            double dist = cv::norm(data.row(i) - centroids.row(j));
            if (dist < min_dist)
            {
                min_dist = dist;
                labels[i] = j;
            }
        }
    }
}

void KMeans::update_centroids(const cv::Mat &data)
{
    cv::Mat new_centroids = cv::Mat::zeros(centroids.size(), centroids.type());
    std::vector<int> counts(k, 0);

    for (int i = 0; i < data.rows; ++i)
    {
        new_centroids.row(labels[i]) += data.row(i);
        counts[labels[i]]++;
    }

    for (int j = 0; j < k; ++j)
    {
        if (counts[j] > 0)
        {
            new_centroids.row(j) /= counts[j];
        }
    }

    centroids = new_centroids;
}