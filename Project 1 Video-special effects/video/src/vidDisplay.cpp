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

        // Apply depth-based filtering
        if (depth_filter)
        {
            for (int i = 0; i < original_frame.rows; i++)
            {
                for (int j = 0; j < original_frame.cols; j++)
                {
                    // Create artistic effect based on depth
                    if (depth_frame.at<unsigned char>(i, j) < 85)
                    {
                        // Close objects - enhance red
                        filter_frame.at<cv::Vec3b>(i, j)[2] = std::min(255, filter_frame.at<cv::Vec3b>(i, j)[2] + 50);
                    }
                    else if (depth_frame.at<unsigned char>(i, j) < 170)
                    {
                        // Mid-range objects - enhance green
                        filter_frame.at<cv::Vec3b>(i, j)[1] = std::min(255, filter_frame.at<cv::Vec3b>(i, j)[1] + 50);
                    }
                    else
                    {
                        // Far objects - enhance blue
                        filter_frame.at<cv::Vec3b>(i, j)[0] = std::min(255, filter_frame.at<cv::Vec3b>(i, j)[0] + 50);
                    }
                }
            }
            cv::imshow("Filter Video", filter_frame);
        }

        // Handle existing filter keys (referenced from original vidDisplay.cpp)
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
    }

    delete capdev;
    return 0;
}