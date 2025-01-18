/*
    Modified to include depth detection
    CS5330 - Pattern Recognition & Computer Vision
*/
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <cstdio>
#include "filter.h"
#include "faceDetect.h"
#include "DA2Network.hpp"
#include <vector>

int main(int argc, char *argv[])
{
    cv::VideoCapture *capdev;
    cv::Mat original_frame;
    cv::Mat filter_frame;
    cv::Mat dst;
    cv::Mat depth_frame;
    cv::Mat depth_vis;
    const float reduction = 0.5;

    // Initialize the DA2Network
    DA2Network da_net("../src/model_fp16.onnx");

    // open the video device
    capdev = new cv::VideoCapture(0);
    if (!capdev->isOpened())
    {
        printf("Unable to open video device\n");
        return -1;
    }

    // get properties of the image
    cv::Size refS((int)capdev->get(cv::CAP_PROP_FRAME_WIDTH),
                  (int)capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
    printf("Expected size: %d %d\n", refS.width, refS.height);

    float scale_factor = 256.0 / (refS.height * reduction);
    printf("Using scale factor %.2f\n", scale_factor);

    cv::namedWindow("Original Video", 1);
    cv::namedWindow("Filter Video", 1);
    cv::namedWindow("Depth", 1);

    int key = 0;
    bool show_depth = false;
    bool depth_filter = false;
    bool grey_face = false;
    bool blur_outside_face = false;
    bool fog_effect = false;

    // Initialize face detector
    cv::CascadeClassifier face_cascade;
    face_cascade.load("../src/haarcascade_frontalface_alt2.xml");

    for (;;)
    {
        *capdev >> original_frame;

        if (original_frame.empty())
        {
            printf("frame is empty\n");
            break;
        }

        // Resize frame for speed
        cv::resize(original_frame, original_frame, cv::Size(), reduction, reduction);
        original_frame.copyTo(filter_frame);

        // Get depth information
        da_net.set_input(original_frame, scale_factor);
        da_net.run_network(depth_frame, original_frame.size());
        cv::applyColorMap(depth_frame, depth_vis, cv::COLORMAP_INFERNO);

        // Always show original video
        cv::imshow("Original Video", original_frame);

        // Handle depth visualization
        if (show_depth)
        {
            cv::imshow("Depth", depth_vis);
        }

        // Store original frame for face detection
        cv::Mat working_frame = filter_frame.clone();
        std::vector<cv::Rect> faces;
        face_cascade.detectMultiScale(working_frame, faces);

        // Grey background with colorful faces
        if (grey_face && !faces.empty())
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
        if (blur_outside_face && !faces.empty())
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
        if (fog_effect)
        {
            for (int i = 0; i < original_frame.rows; i++)
            {
                for (int j = 0; j < original_frame.cols; j++)
                {
                    float depth_val = depth_frame.at<unsigned char>(i, j) / 255.0f;
                    float fog_factor = std::exp(-depth_val * 2); // Adjust multiplier for fog density

                    // Blend with white based on depth
                    filter_frame.at<cv::Vec3b>(i, j)[0] = cv::saturate_cast<uchar>(
                        filter_frame.at<cv::Vec3b>(i, j)[0] * fog_factor + 255 * (1 - fog_factor));
                    filter_frame.at<cv::Vec3b>(i, j)[1] = cv::saturate_cast<uchar>(
                        filter_frame.at<cv::Vec3b>(i, j)[1] * fog_factor + 255 * (1 - fog_factor));
                    filter_frame.at<cv::Vec3b>(i, j)[2] = cv::saturate_cast<uchar>(
                        filter_frame.at<cv::Vec3b>(i, j)[2] * fog_factor + 255 * (1 - fog_factor));
                }
            }
        }

        // Show filtered frame if any filter is active
        if (depth_filter || grey_face || blur_outside_face || fog_effect)
        {
            cv::imshow("Filter Video", filter_frame);
        }

        // Handle key presses
        key = cv::waitKey(10);

        if (key == 'q')
        {
            break;
        }
        else if (key == 'd')
        {
            show_depth = !show_depth;
            printf("Depth view %s\n", show_depth ? "enabled" : "disabled");
        }
        else if (key == 'a')
        {
            depth_filter = !depth_filter;
            printf("Depth-based artistic filter %s\n", depth_filter ? "enabled" : "disabled");
        }
        else if (key == 's')
        {
            cv::imwrite("depth_image.png", depth_vis);
            if (depth_filter)
            {
                cv::imwrite("depth_filtered.png", filter_frame);
            }
            printf("Images saved\n");
        }
        else if (key == 'g')
        {
            grey_face = !grey_face;
            printf("Grey background with colorful faces %s\n", grey_face ? "enabled" : "disabled");
        }
        else if (key == 'b')
        {
            blur_outside_face = !blur_outside_face;
            printf("Blur outside faces %s\n", blur_outside_face ? "enabled" : "disabled");
        }
        else if (key == 'f')
        {
            fog_effect = !fog_effect;
            printf("Fog effect %s\n", fog_effect ? "enabled" : "disabled");
        }
    }

    delete capdev;
    return 0;
}