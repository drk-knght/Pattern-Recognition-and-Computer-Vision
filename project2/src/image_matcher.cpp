#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>
#include "opencv2/opencv.hpp"
#include "csv_util.h"
#include "feature_utils.h"

struct SSDMatch
{
    char *filename;
    float distance;

    bool operator<(const SSDMatch &other) const
    {
        return distance < other.distance;
    }
};

struct HistogramMatch
{
    char *filename;
    float similarity;

    // descending order sorting
    bool operator<(const HistogramMatch &other) const
    {
        return similarity > other.similarity;
    }
};

int main(int argc, char *argv[])
{
    if (argc != 5)
    {
        printf("Usage: %s <target_image> <feature_database.csv> <num_matches>\n", argv[0]);
        return -1;
    }

    char *target_file = argv[1];
    char *database_file = argv[2];
    int num_matches = atoi(argv[3]);
    char *feature_type = argv[4];

    // Read target image in color
    cv::Mat target = cv::imread(target_file);
    if (target.empty())
    {
        printf("Cannot open target image %s\n", target_file);
        return -1;
    }

    if (strcmp(feature_type, "ssd") == 0)
    {

        // Extract features from target
        std::vector<float> target_features = extractCenterPatch(target);
        if (target_features.empty())
        {
            printf("Failed to extract features from target image\n");
            return -1;
        }

        // Read database
        std::vector<char *> filenames;
        std::vector<std::vector<float>> database_features;
        read_image_data_csv(database_file, filenames, database_features);

        // Calculate distances
        std::vector<SSDMatch> matches;
        for (size_t i = 0; i < filenames.size(); i++)
        {
            float dist = calculateSSD(target_features, database_features[i]);
            matches.push_back({filenames[i], dist});
        }

        // Sort matches
        std::sort(matches.begin(), matches.end());

        // Print top N matches
        printf("Top %d matches for %s:\n", num_matches, target_file);
        for (int i = 0; i < num_matches && i < matches.size(); i++)
        {
            printf("%d. %s (distance: %.5f)\n", i + 1, matches[i].filename, matches[i].distance);
        }
    }
    else if (strcmp(feature_type, "hist") == 0)
    {
        // Extract histogram features from target
        std::vector<float> target_features = extractColorHistogram(target);
        if (target_features.empty())
        {
            printf("Failed to extract features from target image\n");
            return -1;
        }

        // Read database
        std::vector<char *> filenames;
        std::vector<std::vector<float>> database_features;
        if (read_image_data_csv(database_file, filenames, database_features) != 0)
        {
            printf("Failed to read database file\n");
            return -1;
        }

        // Calculate similarities
        std::vector<HistogramMatch> matches;

        for (size_t i = 0; i < filenames.size(); i++)
        {
            float similarity = calculateHistIntersection(target_features, database_features[i]);
            // Read the match image
            HistogramMatch m = {filenames[i], similarity};
            matches.push_back(m);
        }

        // Sort matches (now by similarity in descending order)
        std::sort(matches.begin(), matches.end());

        // Print top N matches
        printf("Top %d matches for %s:\n", num_matches, target_file);
        for (int i = 0; i < num_matches && i < matches.size(); i++)
        {
            // Extract just the filename for display
            std::string filename = matches[i].filename;
            printf("%d. %s (similarity: %.3f)\n", i + 1, matches[i].filename, matches[i].similarity);
        }
    }
    else if (strcmp(feature_type, "spatialhist") == 0)
    {
        // Extract spatial histogram features from target
        std::vector<float> target_features = extractSpatialColorHistogram(target);
        if (target_features.empty())
        {
            printf("Failed to extract features from target image\n");
            return -1;
        }

        // Read database
        std::vector<char *> filenames;
        std::vector<std::vector<float>> database_features;
        if (read_image_data_csv(database_file, filenames, database_features) != 0)
        {
            printf("Failed to read database file\n");
            return -1;
        }

        // Calculate similarities using spatial histogram intersection
        std::vector<HistogramMatch> matches;
        for (size_t i = 0; i < filenames.size(); i++)
        {
            float similarity = calculateSpatialHistIntersection(target_features, database_features[i]);
            matches.push_back({filenames[i], similarity});
        }

        // Sort matches (by similarity in descending order)
        std::sort(matches.begin(), matches.end());

        // Print top N matches
        printf("Top %d matches for %s:\n", num_matches, target_file);
        for (int i = 0; i < num_matches && i < matches.size(); i++)
        {
            printf("%d. %s (similarity: %.3f)\n", i + 1, matches[i].filename, matches[i].similarity);
        }
    }
    else if(strcmp(feature_type, "combined") == 0) {
        // Extract combined features from target
        std::vector<float> target_features = extractCombinedFeatures(target);
        if (target_features.empty()) {
            printf("Failed to extract features from target image\n");
            return -1;
        }
        
        // Read database
        std::vector<char *> filenames;
        std::vector<std::vector<float>> database_features;
        if (read_image_data_csv(database_file, filenames, database_features) != 0) {
            printf("Failed to read database file\n");
            return -1;
        }

        // Calculate similarities using combined distance
        std::vector<HistogramMatch> matches;
        for (size_t i = 0; i < filenames.size(); i++) {
            float similarity = calculateCombinedDistance(target_features, database_features[i]);
            matches.push_back({filenames[i], similarity});
        }

        // Sort matches (by similarity in descending order)
        std::sort(matches.begin(), matches.end());

        // Print top N matches
        printf("Top %d matches for %s:\n", num_matches, target_file);
        for (int i = 0; i < num_matches && i < matches.size(); i++) {
            printf("%d. %s (similarity: %.3f)\n", i + 1, matches[i].filename, matches[i].similarity);
        }
    }
    return 0;
}
