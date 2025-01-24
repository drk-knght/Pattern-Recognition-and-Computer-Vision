/*
    Agnibha Chatterjee
    Om Agarwal
    Jan 12 2024
    CS5330- Pattern Recognition & Computer Vision
    This file is the entry
*/
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <cstdio>
#include "filter.h"
#include "faceDetect.h"
#include "DA2Network.hpp"
#include <vector>
#include <ctime>

int main(int argc, char *argv[])
{
    // Pointer to the video capture device
    cv::VideoCapture *capdev;
    // Matrices to hold frames
    cv::Mat original_frame, filter_frame, dst, depth_frame, depth_vis;
    const float reduction = 0.5; // Scale factor for frame resizing

    // Initialize the DA2Network with the model file
    DA2Network da_net("../src/model_fp16.onnx");

    // Open the video device (camera)
    capdev = new cv::VideoCapture(0);
    if (!capdev->isOpened())
    {
        printf("Unable to open video device\n");
        return -1; // Exit if the camera cannot be opened
    }

    // Get properties of the video frame
    cv::Size refS((int)capdev->get(cv::CAP_PROP_FRAME_WIDTH),
                  (int)capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
    printf("Expected size: %d %d\n", refS.width, refS.height);

    // Calculate scale factor for resizing frames
    float scale_factor = 256.0 / (refS.height * reduction);
    printf("Using scale factor %.2f\n", scale_factor);

    // Create windows to display video frames
    cv::namedWindow("Original Video", 1);
    cv::namedWindow("Filter Video", 1);
    cv::namedWindow("Depth", 1);

    // Initialize key variables for filter options
    int key = 0;
    bool show_depth = false;        // Flag to show depth visualization
    bool depth_filter = false;      // Flag for depth-based filtering
    bool grey_face = false;         // Flag for grey background with colorful faces
    bool blur_outside_face = false; // Flag to blur outside faces
    bool fog_effect = false;        // Flag for fog effect

    // Initialize filter keys
    int opencv_grey_key = 0;
    int custom_grey_key = 0;
    int sepia_key = 0;
    int blur_key = 0;
    int sobel_x_key = 0;
    int sobel_y_key = 0;
    int blur_quantize_key = 0;

    // New filter keys
    bool isolate_red = false;     // Flag for isolating red colors
    bool negative_filter = false; // Flag for negative filter

    // Initialize face detector
    cv::CascadeClassifier face_cascade;
    face_cascade.load("../src/haarcascade_frontalface_alt2.xml");

    // Initialize recording variables
    bool is_recording = false;                         // Flag to indicate if recording is active
    cv::VideoWriter video_writer;                      // Video writer object for saving video
    int frame_width = refS.width * reduction;          // Width of the video frame
    int frame_height = refS.height * reduction;        // Height of the video frame
    double fps = 30.0;                                 // Default frames per second
    double device_fps = capdev->get(cv::CAP_PROP_FPS); // Get device FPS
    if (device_fps > 0 && device_fps <= 60.0)
    { // Use device FPS if it's reasonable
        fps = device_fps;
    }
    int frame_count = 0;                                // Count of frames recorded
    const int MAX_FRAMES = static_cast<int>(5.0 * fps); // Maximum frames for 5 seconds

    // Initialize caption text and text display parameters
    std::string caption_text;
    const int TEXT_MARGIN_TOP = 50;             // Margin from the top for text
    const double FONT_SCALE = 1.0;              // Scale for font size
    const int FONT_THICKNESS = 2;               // Thickness of the font
    const cv::Scalar TEXT_COLOR(255, 255, 255); // Color for text (white)

    for (;;)
    {
        // Capture a new frame from the video device
        *capdev >> original_frame;

        // Check if the frame is empty
        if (original_frame.empty())
        {
            printf("frame is empty\n");
            break; // Exit loop if no frame is captured
        }

        // Resize the frame for processing speed
        cv::resize(original_frame, original_frame, cv::Size(), reduction, reduction);
        original_frame.copyTo(filter_frame); // Copy original frame to filter frame

        // Get depth information from the network
        da_net.set_input(original_frame, scale_factor);
        da_net.run_network(depth_frame, original_frame.size());
        cv::applyColorMap(depth_frame, depth_vis, cv::COLORMAP_INFERNO); // Apply color map to depth frame

        // Store original frame for face detection
        cv::Mat working_frame = filter_frame.clone();
        std::vector<cv::Rect> faces;                         // Vector to hold detected faces
        face_cascade.detectMultiScale(working_frame, faces); // Detect faces in the frame

        // Apply filters based on user input
        // Check for each filter key and apply the corresponding filter
        // opencv default grayscale image function for the video frames
        if (opencv_grey_key)
        {
            cv::cvtColor(original_frame, filter_frame, cv::COLOR_BGR2GRAY);
            cv::imshow("Filter Video", filter_frame);
        }

        // custom grayscale filter for the video frames
        else if (custom_grey_key)
        {
            greyscale(original_frame, filter_frame);
            cv::imshow("Filter Video", filter_frame);
        }

        // sepia image filter for the video frames
        else if (sepia_key)
        {
            sepia(original_frame, filter_frame);
            cv::imshow("Filter Video", filter_frame);
        }

        // blur image filter for the video frames
        else if (blur_key)
        {
            blur5x5_2(original_frame, filter_frame);
            cv::imshow("Filter Video", filter_frame);
        }

        // sobel x image filter to detect vertical edges
        else if (sobel_x_key)
        {
            sobelX3x3(original_frame, filter_frame);

            // Convert to CV_8U and scale for proper display
            filter_frame.convertTo(filter_frame, CV_8U, 0.5, 128);
            cv::imshow("Filter Video", filter_frame);
        }

        // sobel y image filter to detect horizontal edges
        else if (sobel_y_key)
        {
            sobelY3x3(original_frame, filter_frame);

            // Convert to CV_8U and scale for proper display
            filter_frame.convertTo(filter_frame, CV_8U, 0.5, 128);
            cv::imshow("Filter Video", filter_frame);
        }

        // blur quantize filter
        else if (blur_quantize_key)
        {
            blurQuantize(original_frame, filter_frame, 10);
            cv::imshow("Filter Video", filter_frame);
        }

        // Isolate red color to get only red pigments of the image
        else if (isolate_red)
        {
            isolateRed(original_frame, filter_frame);
        }

        // Negative filter to negate the image
        else if (negative_filter)
        {
            filter_frame = cv::Scalar(255, 255, 255) - filter_frame;
        }

        // Grey background with colorful faces
        else if (grey_face && !faces.empty())
        {
            cv::Mat grey;
            cv::cvtColor(filter_frame, grey, cv::COLOR_BGR2GRAY);
            cv::cvtColor(grey, filter_frame, cv::COLOR_GRAY2BGR);

            // Keep faces colorful
            for (const auto &face : faces)
            {
                working_frame(face).copyTo(filter_frame(face));
            }
        }

        // Blur outside faces
        else if (blur_outside_face && !faces.empty())
        {
            cv::Mat blurred;
            cv::GaussianBlur(filter_frame, blurred, cv::Size(25, 25), 0);
            blurred.copyTo(filter_frame);

            // Restore unblurred faces
            for (const auto &face : faces)
            {
                working_frame(face).copyTo(filter_frame(face));
            }
        }

        // Fog effect using depth
        else if (fog_effect)
        {
            for (int i = 0; i < original_frame.rows; i++)
            {
                for (int j = 0; j < original_frame.cols; j++)
                {
                    float depth_val = depth_frame.at<unsigned char>(i, j) / 255.0f;
                    float fog_factor = std::exp(-depth_val * 2);

                    filter_frame.at<cv::Vec3b>(i, j)[0] = cv::saturate_cast<uchar>(
                        filter_frame.at<cv::Vec3b>(i, j)[0] * fog_factor + 255 * (1 - fog_factor));
                    filter_frame.at<cv::Vec3b>(i, j)[1] = cv::saturate_cast<uchar>(
                        filter_frame.at<cv::Vec3b>(i, j)[1] * fog_factor + 255 * (1 - fog_factor));
                    filter_frame.at<cv::Vec3b>(i, j)[2] = cv::saturate_cast<uchar>(
                        filter_frame.at<cv::Vec3b>(i, j)[2] * fog_factor + 255 * (1 - fog_factor));
                }
            }
        }

        // Always show the original video
        cv::imshow("Original Video", original_frame);

        // Show depth visualization if enabled
        if (show_depth)
        {
            cv::imshow("Depth", depth_vis);
        }

        // Add text overlay to the frame if caption exists
        if (!caption_text.empty())
        {
            // Calculate text size to center it
            int baseline = 0;
            cv::Size text_size = cv::getTextSize(caption_text, cv::FONT_HERSHEY_DUPLEX,
                                                 FONT_SCALE, FONT_THICKNESS, &baseline);

            cv::Point text_org((filter_frame.cols - text_size.width) / 2, // Center horizontally
                               TEXT_MARGIN_TOP + text_size.height);       // Top margin

            // Draw black outline for better visibility
            cv::putText(filter_frame, caption_text, text_org,
                        cv::FONT_HERSHEY_DUPLEX, FONT_SCALE,
                        cv::Scalar(0, 0, 0), FONT_THICKNESS + 1); // Thicker black outline

            // Draw white text
            cv::putText(filter_frame, caption_text, text_org,
                        cv::FONT_HERSHEY_DUPLEX, FONT_SCALE,
                        TEXT_COLOR, FONT_THICKNESS);
        }

        // Show filtered frame if any filter is active
        if (opencv_grey_key || custom_grey_key || sepia_key || blur_key ||
            sobel_x_key || sobel_y_key || blur_quantize_key ||
            depth_filter || grey_face || blur_outside_face || fog_effect ||
            isolate_red || negative_filter || !caption_text.empty()) // Added condition for caption
        {
            cv::imshow("Filter Video", filter_frame);
        }

        // Handle recording logic
        if (is_recording)
        {
            if (frame_count >= MAX_FRAMES)
            {
                // Stop recording after 5 seconds worth of frames
                is_recording = false;
                video_writer.release(); // Release the video writer
                printf("Recording stopped - 5 second limit reached\n");
            }
            else
            {
                // Write the frame with applied filters
                video_writer.write(filter_frame);                           // This will include the text overlay
                frame_count++;                                              // Increment frame count
                printf("Recording frame %d/%d\n", frame_count, MAX_FRAMES); // Added progress indicator
            }
        }

        // Handle key presses for user input
        key = cv::waitKey(10); // Wait for a key press for 10 ms

        // Check for specific key presses to control the application
        if (key == 'q')
        {
            if (is_recording)
            {
                video_writer.release();
            }
            break;
        }
        else if (key == 'd')
        {
            show_depth = !show_depth;
            printf("Depth view %s\n", show_depth ? "enabled" : "disabled");
        }
        else if (key == 'k')
        {
            grey_face = !grey_face;
            printf("Grey background with colorful faces %s\n", grey_face ? "enabled" : "disabled");
        }
        else if (key == 'v')
        {
            fog_effect = !fog_effect;
            printf("Fog effect %s\n", fog_effect ? "enabled" : "disabled");
        }
        else if (key == 'g')
        {
            opencv_grey_key += 1;
            custom_grey_key = 0;
            sepia_key = 0;
            blur_key = 0;
            sobel_x_key = 0;
            sobel_y_key = 0;
            blur_quantize_key = 0;
        }
        else if (key == 'h')
        {
            custom_grey_key += 1;
            opencv_grey_key = 0;
            sepia_key = 0;
            blur_key = 0;
            sobel_x_key = 0;
            sobel_y_key = 0;
            blur_quantize_key = 0;
        }

        // update the sepia filter key for the video frames
        else if (key == 'e')
        {
            sepia_key += 1;
            opencv_grey_key = 0;
            custom_grey_key = 0;
            blur_key = 0;
            sobel_x_key = 0;
            sobel_y_key = 0;
            blur_quantize_key = 0;
        }

        // update the blur filter key for the video frames
        else if (key == 'b')
        {
            blur_key += 1;
            opencv_grey_key = 0;
            custom_grey_key = 0;
            sepia_key = 0;
            sobel_x_key = 0;
            sobel_y_key = 0;
            blur_quantize_key = 0;
        }

        // update the sobel x filter key for the video frames
        else if (key == 'x')
        {
            printf("Sobel x filter enabled\n");
            blur_key = 0;
            opencv_grey_key = 0;
            custom_grey_key = 0;
            sepia_key = 0;
            sobel_y_key = 0;
            blur_quantize_key = 0;
            sobel_x_key += 1;
        }

        // update the sobel y filter key for the video frames
        else if (key == 'y')
        {
            printf("Sobel y filter enabled\n");
            blur_key = 0;
            opencv_grey_key = 0;
            custom_grey_key = 0;
            sepia_key = 0;
            sobel_x_key = 0;
            blur_quantize_key = 0;
            sobel_y_key += 1;
        }

        // update the blur quantization filter key for the video frames
        else if (key == 'l')
        {
            blur_quantize_key = 1;
            blur_key = 0;
            opencv_grey_key = 0;
            custom_grey_key = 0;
            sepia_key = 0;
            sobel_y_key = 0;
            sobel_x_key = 0;
        }

        else if (key == 'z')
        {
            blur_outside_face = !blur_outside_face;
            printf("Blur outside faces %s\n", blur_outside_face ? "enabled" : "disabled");
        }
        else if (key == 'a')
        {
            depth_filter = !depth_filter;
            printf("Depth-based artistic filter %s\n", depth_filter ? "enabled" : "disabled");
        }
        else if (key == 's')
        {
            cv::imwrite("../images/original.png", original_frame);
            cv::imwrite("../images/filter.png", filter_frame);
            // if (depth_filter)
            // {
            //     cv::imwrite("depth_filtered.png", filter_frame);
            // }
            // printf("Images saved\n");
        }
        else if (key == 'i')
        {
            isolate_red = !isolate_red;
            printf("Red isolation filter %s\n", isolate_red ? "enabled" : "disabled");
        }
        else if (key == 'n')
        {
            negative_filter = !negative_filter;
            printf("Negative filter %s\n", negative_filter ? "enabled" : "disabled");
        }
        else if (key == 'r')
        {
            if (!is_recording)
            {
                // Start recording
                std::string filename = "output_" + std::to_string(time(nullptr)) + ".avi";
                video_writer.open(filename, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                                  fps, cv::Size(frame_width, frame_height));

                if (!video_writer.isOpened())
                {
                    printf("Failed to create video file\n");
                    continue;
                }

                // Get caption text from user
                printf("Enter caption text (press Enter when done): ");
                std::getline(std::cin, caption_text);

                is_recording = true;
                frame_count = 0;
                printf("Started recording to %s (FPS: %.2f)\n", filename.c_str(), fps);
            }
            else
            {
                // Stop recording
                is_recording = false;
                video_writer.release();
                printf("Recording stopped after %d frames\n", frame_count);
                caption_text.clear(); // Clear the caption for next recording
            }
        }
    }

    delete capdev; // Clean up the video capture device
    return 0;      // Exit the program
}