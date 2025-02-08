/*
    Agnibha Chatterjee
    Om Agarwal
    Feb 8 2025
    CS5330- Pattern Recognition & Computer Vision
    This file is the entry point for question 5 of the assignment.
*/

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <iomanip>
#include "feature_utils.h"
#include "csv_util.h"

// Forward declaration for the new combined DNN + spatial variance function.
// (You may also add this to feature_utils.h.)
std::vector<float> extractCombinedDnnSpatialVarianceFeatures(const cv::Mat &img);

// A simple structure to hold a match result.
struct Match
{
    std::string filename;
    double distance;
};

// Example distance functions.

// SSD distance for raw pixels or simple feature vectors.
double computeSSD(const std::vector<float> &v1, const std::vector<float> &v2)
{
    if (v1.size() != v2.size())
        return std::numeric_limits<double>::max();
    double ssd = 0.0;
    for (size_t i = 0; i < v1.size(); i++)
    {
        double diff = v1[i] - v2[i];
        ssd += diff * diff;
    }
    return ssd;
}

// Histogram intersection distance (normalized so that higher intersection is more similar).
double histogramIntersectionDistance(const std::vector<float> &h1, const std::vector<float> &h2)
{
    if (h1.size() != h2.size())
        return 1.0; // maximum distance if different sizes
    double intersection = 0.0;
    for (size_t i = 0; i < h1.size(); i++)
    {
        intersection += std::min(h1[i], h2[i]);
    }
    // Return a distance (1 - intersection) so that lower is better.
    return 1.0 - intersection;
}

int main(int argc, char *argv[])
{
    // Updated usage:
    // For method "dnndsv", only 3 arguments are required.
    // Otherwise, the usage is: <target_image> <database_directory> <method> <top_N>
    if (argc < 4)
    {
        std::cout << "Usage: " << argv[0]
                  << " <target_image> <database_directory> <method> [<top_N>]" << std::endl;
        std::cout << "Methods: baseline, hist, spatialhist, combined, orb, lbp, ssim, spatialvar, dnndsv or '--compare-all'" << std::endl;
        std::cout << "Note: When using method 'dnndsv', do not pass a top_N parameter; all matches will be returned." << std::endl;
        return -1;
    }

    std::string targetImagePath = argv[1];
    std::string dbDir = argv[2];
    std::string methodArg = argv[3];
    int topN = 0;

    // If method is "dnndsv", no top_N argument should be provided.
    if (methodArg == "dnndsv")
    {
        if (argc != 4)
        {
            std::cerr << "Error: When using method 'dnndsv', do not provide a top_N count." << std::endl;
            std::cout << "Usage: " << argv[0] << " <target_image> <database_directory> dnndsv" << std::endl;
            return -1;
        }
        topN = -1; // Flag value to indicate no top-N limitation.
    }
    else
    {
        if (argc != 5)
        {
            std::cerr << "Error: Incorrect number of arguments." << std::endl;
            std::cout << "Usage: " << argv[0] << " <target_image> <database_directory> <method> <top_N>" << std::endl;
            return -1;
        }
        topN = std::stoi(argv[4]);
    }

    // Read the target image.
    cv::Mat targetImg = cv::imread(targetImagePath);
    if (targetImg.empty())
    {
        std::cerr << "Error: Cannot open target image " << targetImagePath << std::endl;
        return -1;
    }

    // Discover images in the database directory.
    std::vector<cv::String> imageFiles;
    cv::glob(dbDir + "/*.*", imageFiles, false);
    if (imageFiles.empty())
    {
        std::cerr << "No images found in directory " << dbDir << std::endl;
        return -1;
    }

    // If the method parameter is "all" (or "--compare-all"), run all techniques.
    if (methodArg == "all" || methodArg == "--compare-all")
    {
        // Add the new dnndsv method to the list.
        std::vector<std::string> methods = {"baseline", "hist", "spatialhist", "combined", "orb", "lbp", "ssim", "spatialvar", "dnndsv"};

        // Print table header.
        std::cout << std::left << std::setw(15) << "Method"
                  << std::setw(6) << "Rank"
                  << std::setw(40) << "Filename"
                  << std::setw(12) << "Distance" << std::endl;
        std::cout << std::string(75, '-') << std::endl;

        // For each method, compute matches and output the results.
        for (const auto &m : methods)
        {
            std::vector<float> targetFeatures;
            if (m == "baseline")
                targetFeatures = extractCenterPatch(targetImg);
            else if (m == "hist")
                targetFeatures = extractColorHistogram(targetImg);
            else if (m == "spatialhist")
                targetFeatures = extractSpatialColorHistogram(targetImg);
            else if (m == "combined")
                targetFeatures = extractCombinedFeatures(targetImg);
            else if (m == "orb")
                targetFeatures = extractORBFeatures(targetImg);
            else if (m == "lbp")
                targetFeatures = extractLBPFeatures(targetImg);
            else if (m == "ssim")
                targetFeatures = extractSSIMFeatures(targetImg);
            else if (m == "spatialvar")
                targetFeatures = extractColorSpatialVariance(targetImg);
            // New branch for the combined DNN + spatial variance method.
            else if (m == "dnndsv")
                targetFeatures = extractCombinedDnnSpatialVarianceFeatures(targetImg);

            // Loop over all database images.
            std::vector<Match> matches;
            for (const auto &file : imageFiles)
            {
                // Skip the target image if in the same directory.
                if (file.find(targetImagePath) != std::string::npos)
                    continue;
                cv::Mat img = cv::imread(file);
                if (img.empty())
                    continue;

                std::vector<float> features;
                if (m == "baseline")
                    features = extractCenterPatch(img);
                else if (m == "hist")
                    features = extractColorHistogram(img);
                else if (m == "spatialhist")
                    features = extractSpatialColorHistogram(img);
                else if (m == "combined")
                    features = extractCombinedFeatures(img);
                else if (m == "orb")
                    features = extractORBFeatures(img);
                else if (m == "lbp")
                    features = extractLBPFeatures(img);
                else if (m == "ssim")
                    features = extractSSIMFeatures(img);
                else if (m == "spatialvar")
                    features = extractColorSpatialVariance(img);
                else if (m == "dnndsv")
                    features = extractCombinedDnnSpatialVarianceFeatures(img);

                double d = 0.0;
                // Use computeSSD for these methods.
                if (m == "baseline" || m == "orb" || m == "dnndsv")
                    d = computeSSD(targetFeatures, features);
                else if (m == "spatialvar")
                    d = calculateSpatialVarianceDistance(targetFeatures, features);
                else
                    d = histogramIntersectionDistance(targetFeatures, features);

                matches.push_back({file, d});
            }

            // Sort matches (smaller distance indicates higher similarity).
            std::sort(matches.begin(), matches.end(), [](const Match &a, const Match &b)
                      { return a.distance < b.distance; });

            // For methods other than "dnndsv", limit to topN matches.
            // For dnndsv, we return all matching images.
            size_t outputCount = matches.size();
            if (m != "dnndsv" && matches.size() > static_cast<size_t>(topN))
                outputCount = topN;

            // Output the results for this method.
            for (size_t i = 0; i < outputCount; i++)
            {
                std::cout << std::left << std::setw(15) << m
                          << std::setw(6) << (i + 1)
                          << std::setw(40) << matches[i].filename
                          << std::setw(12) << matches[i].distance
                          << std::endl;
            }
            std::cout << std::string(75, '-') << std::endl;
        }
        return 0;
    }

    // Else, run a single matching method using the provided feature type.
    std::vector<float> targetFeatures;
    if (methodArg == "baseline")
        targetFeatures = extractCenterPatch(targetImg);
    else if (methodArg == "hist")
        targetFeatures = extractColorHistogram(targetImg);
    else if (methodArg == "spatialhist")
        targetFeatures = extractSpatialColorHistogram(targetImg);
    else if (methodArg == "combined")
        targetFeatures = extractCombinedFeatures(targetImg);
    else if (methodArg == "orb")
        targetFeatures = extractORBFeatures(targetImg);
    else if (methodArg == "lbp")
        targetFeatures = extractLBPFeatures(targetImg);
    else if (methodArg == "ssim")
        targetFeatures = extractSSIMFeatures(targetImg);
    else if (methodArg == "spatialvar")
        targetFeatures = extractColorSpatialVariance(targetImg);
    else if (methodArg == "dnndsv")
        targetFeatures = extractCombinedDnnSpatialVarianceFeatures(targetImg);
    else
    {
        std::cerr << "Unknown method: " << methodArg << std::endl;
        return -1;
    }

    std::vector<Match> matches;
    for (const auto &file : imageFiles)
    {
        if (file.find(targetImagePath) != std::string::npos)
            continue;
        cv::Mat img = cv::imread(file);
        if (img.empty())
            continue;

        std::vector<float> features;
        if (methodArg == "baseline")
            features = extractCenterPatch(img);
        else if (methodArg == "hist")
            features = extractColorHistogram(img);
        else if (methodArg == "spatialhist")
            features = extractSpatialColorHistogram(img);
        else if (methodArg == "combined")
            features = extractCombinedFeatures(img);
        else if (methodArg == "orb")
            features = extractORBFeatures(img);
        else if (methodArg == "lbp")
            features = extractLBPFeatures(img);
        else if (methodArg == "ssim")
            features = extractSSIMFeatures(img);
        else if (methodArg == "spatialvar")
            features = extractColorSpatialVariance(img);
        else if (methodArg == "dnndsv")
            features = extractCombinedDnnSpatialVarianceFeatures(img);

        double d = 0.0;
        if (methodArg == "baseline" || methodArg == "orb" || methodArg == "dnndsv")
            d = computeSSD(targetFeatures, features);
        else if (methodArg == "spatialvar")
            d = calculateSpatialVarianceDistance(targetFeatures, features);
        else
            d = histogramIntersectionDistance(targetFeatures, features);

        matches.push_back({file, d});
    }

    // Sort matches (smaller distance means more similar).
    std::sort(matches.begin(), matches.end(), [](const Match &a, const Match &b)
              { return a.distance < b.distance; });

    // For methods other than dnndsv, limit to topN.
    size_t outputCount = matches.size();
    if (methodArg != "dnndsv" && matches.size() > static_cast<size_t>(topN))
        outputCount = topN;

    // Print the results.
    std::cout << "Top " << outputCount << " matches using method '" << methodArg << "':" << std::endl;
    for (size_t i = 0; i < outputCount; i++)
    {
        std::cout << (i + 1) << ". " << matches[i].filename
                  << " (distance = " << matches[i].distance << ")" << std::endl;
    }
    return 0;
}