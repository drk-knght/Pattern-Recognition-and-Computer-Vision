#include "ImageProcessor.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <dirent.h> // Include for directory handling
#include <string>

int main()
{
    ImageProcessor processor;                           // Create an instance of ImageProcessor
    std::string directory_path = "../input_directory/"; // Specify the directory containing images

    // Open the directory
    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir(directory_path.c_str())) != nullptr)
    {
        while ((ent = readdir(dir)) != nullptr)
        {
            std::string filename = ent->d_name;

            // Skip the current and parent directory entries
            if (filename == "." || filename == "..")
                continue;

            std::string full_path = directory_path + filename;
            cv::Mat frame = cv::imread(full_path); // Read each image

            if (frame.empty())
                continue; // Skip if the image is empty

            cv::Mat threshold_frame;                         // Declare a variable to hold the threshold frame
            processor.process_frame(frame, threshold_frame); // Process the frame

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

    processor.destroy_windows();
    return 0;
}