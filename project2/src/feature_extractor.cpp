/*
    Agnibha Chatterjee
    Om Agarwal
    Feb 8 2025
    CS5330- Pattern Recognition & Computer Vision
    This file extracts features from images in a directory and writes them to a CSV file using a specified feature extraction method.
*/
#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include "csv_util.h"
#include "feature_utils.h"

int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        printf("Usage: %s <directory_path> <output_file_path> <feature_type>\n", argv[0]);
        printf("feature_type options: 'ssd' or 'hist'\n");
        exit(-1);
    }

    char *dirname = argv[1];
    char *output_file = argv[2];
    char *feature_type = argv[3];

    DIR *dirp = opendir(dirname);
    if (!dirp)
    {
        printf("Cannot open directory %s\n", dirname);
        exit(-1);
    }

    struct dirent *dp;
    bool first_file = true;

    // Process each image in directory
    while ((dp = readdir(dirp)) != NULL)
    {
        // Check for image files (you can add more extensions if needed)
        if (strstr(dp->d_name, ".jpg"))
        {
            char buffer[256];
            sprintf(buffer, "%s/%s", dirname, dp->d_name);

            // Read image in color
            cv::Mat img = cv::imread(buffer);
            if (img.empty())
            {
                printf("Failed to read image: %s\n", buffer);
                continue;
            }

            std::vector<float> features;

            // Extract features based on type
            if (strcmp(feature_type, "ssd") == 0)
            {
                features = extractCenterPatch(img);
            }
            else if (strcmp(feature_type, "hist") == 0)
            {
                features = extractColorHistogram(img);
            }
            else if (strcmp(feature_type, "spatialhist") == 0)
            {
                features = extractSpatialColorHistogram(img);
            }
            else if (strcmp(feature_type, "combined") == 0)
            {
                features = extractCombinedFeatures(img);
            }
            else
            {
                printf("Unknown feature type: %s\n", feature_type);
                closedir(dirp);
                exit(-1);
            }

            // Save to CSV
            append_image_data_csv(output_file, dp->d_name, features, first_file);
            first_file = false;
        }
    }

    printf("Feature extraction complete. Results saved to: %s\n", output_file);
    closedir(dirp);
    return 0;
}