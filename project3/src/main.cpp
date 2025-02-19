#include "ImageProcessor.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <dirent.h> // Include for directory handling
#include <string>

int main(int argc, char **argv)
{
    // Check if a directory path was provided as an argument
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <directory_path> [train]" << std::endl;
        return -1;
    }

    // If a second argument (e.g., "train") is provided, enable training mode.
    bool training_mode = false;
    if (argc >= 3)
    {
        std::string mode = argv[2];
        if (mode == "train")
        {
            training_mode = true;
            std::cout << "Training mode enabled. Press 't' to save training data for each image." << std::endl;
        }
    }

    ImageProcessor processor;             // Create an instance of ImageProcessor
    std::string directory_path = argv[1]; // Use the provided directory path

    // Optionally, add a trailing slash if not present (covers both Unix and Windows)
    if (directory_path.back() != '/' && directory_path.back() != '\\')
        directory_path.push_back('/');

    // Open the directory
    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir(directory_path.c_str())) != nullptr)
    {
        printf("Processing images in directory: %s\n", directory_path.c_str());
        while ((ent = readdir(dir)) != nullptr)
        {
            std::string filename = ent->d_name;

            // Skip the current and parent directory entries
            if (filename == "." || filename == "..")
                continue;

            // Skip files that already have "_threshold" in their name
            if (filename.find("_threshold") != std::string::npos)
                continue;

            std::string full_path = directory_path + filename;
            cv::Mat frame = cv::imread(full_path); // Read each image

            if (frame.empty())
                continue; // Skip if the image is empty

            cv::Mat threshold_frame;                         // Declare a variable to hold the threshold frame
            processor.process_frame(frame, threshold_frame); // Process the frame

            // If in training mode, wait for a key press—if 't' is pressed, save the training data.
            if (training_mode)
            {
                int key = cv::waitKey(0);
                if (key == 't' || key == 'T')
                {
                    processor.collect_training_data(processor.get_threshold_frame());
                }
            }
            else
            {
                cv::waitKey(1);
            }

            // Save the original and threshold frames back to the same directory
            cv::imwrite(full_path, frame);                                                                                    // Save the original frame
            cv::imwrite(directory_path + filename.substr(0, filename.find_last_of('.')) + "_threshold.jpg", threshold_frame); // Save the threshold frame
        }
        closedir(dir); // Close the directory
    }
    else
    {
        std::cerr << "Could not open directory: " << directory_path << std::endl;
        return -1; // Return an error code if the directory cannot be opened
    }

    printf("Processing complete. Please check the %s\n", directory_path.c_str());
    processor.destroy_windows();
    return 0;
}