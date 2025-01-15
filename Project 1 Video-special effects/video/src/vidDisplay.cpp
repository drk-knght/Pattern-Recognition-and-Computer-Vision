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

int main(int argc, char *argv[]){
    cv::VideoCapture *capdev;

    //open the video device
    capdev=new cv::VideoCapture(0);
    
    // Error occured while opening the video capture
    if(!capdev->isOpened()){
        printf("Unable to open the video device\n");
        return -1;
    }

    // get some properties of the image
    cv::Size refS((int)capdev->get(cv::CAP_PROP_FRAME_WIDTH),(int)capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
    printf("Expected size: %d %d\n",refS.width,refS.height);

    // cv::namedWindow("Video",1); //indetifies a window
    cv::Mat original_frame;
    cv::Mat filter_frame;
    cv::Mat dst;

    int key=0;
    int opencv_grey_key=0;
    int custom_grey_key=0;
    int sepia_key=0;
    int blur_key=0;
    int sobel_x_key=0;
    int sobel_y_key=0;
    int blur_quantize_key=0;

    for(;;){
        *capdev>>original_frame; //get a new frame from the camera, treat as a stream
        
        // check if the frame is empty
        if(original_frame.empty()){
            printf("frame is empty\n");
            break;
        }

        original_frame.copyTo(filter_frame);

        // display the video frame
        cv::imshow("Original Video",original_frame);

        // opencv default grayscale image function for the video frames
        if(opencv_grey_key){
            cv::cvtColor(original_frame, filter_frame, cv::COLOR_BGRA2GRAY);
            cv::imshow("Filter Video",filter_frame);
        }

        // custom grayscale filter for the video frames
        else if(custom_grey_key){
            greyscale(original_frame,filter_frame);
            cv::imshow("Filter Video",filter_frame);
        }

        // sepia image filter for the video frames
        else if(sepia_key){
            sepia(original_frame,filter_frame);
            // dst.copyTo(frame);
            cv::imshow("Filter Video",filter_frame);
        }

        // blur image filter for the video frames
        else if(blur_key){
            blur5x5_2(original_frame,filter_frame);
            // dst.copyTo(frame);
            cv::imshow("Filter Video",filter_frame);
        }

        // sobel x image filter to detect vertical edges
        else if(sobel_x_key){
            sobelX3x3(original_frame,filter_frame);

            // converting the pixel values to abs values
            cv::Mat abs_filter_frame;
            cv::convertScaleAbs(filter_frame,abs_filter_frame);
            
            cv::imshow("Filter Video",abs_filter_frame);
        }        
        
        // sobel y image filter to detect horizontal edges
        else if(sobel_y_key){
            sobelY3x3(original_frame,filter_frame);

            // converting the pixel values to abs values
            cv::Mat abs_filter_frame;
            cv::convertScaleAbs(filter_frame,abs_filter_frame);

            cv::imshow("Filter Video",abs_filter_frame);
        }


        else if(blur_quantize_key){
            blurQuantize(original_frame,filter_frame,10);
            cv::imshow("Filter Video",filter_frame);
        }

        // see if there is a waiting keystroke
        key=cv::waitKey(10);

        // update the last pressed key for the required filter opertion
        // Saving the image
        if(key=='s'){
            cv::imwrite("../filter_frame.jpeg",filter_frame);
            cv::imwrite("../original_frame.jpeg",original_frame);
        }

        // update the opencv grey filter key for the video frames
        else if(key=='g'){
            opencv_grey_key+=1;
            custom_grey_key=0; sepia_key=0;
            blur_key=0; sobel_x_key=0;
            sobel_y_key=0; blur_quantize_key=0;

        }

        // update the custom greyscale filter key for the video frames
        else if(key=='h'){
            custom_grey_key+=1;
            opencv_grey_key=0; sepia_key=0;
            blur_key=0; sobel_x_key=0;
            sobel_y_key=0; blur_quantize_key=0;
        }

        // update the sepia filter key for the video frames
        else if(key=='e'){
            sepia_key+=1;
            opencv_grey_key=0; custom_grey_key=0;
            blur_key=0; sobel_x_key=0;
            sobel_y_key=0; blur_quantize_key=0;
        }

        // update the blur filter key for the video frames
        else if(key=='b'){
            blur_key+=1;
            opencv_grey_key=0; custom_grey_key=0;
            sepia_key=0; sobel_x_key=0;
            sobel_y_key=0; blur_quantize_key=0;
        }

        // update the sobel x filter key for the video frames
        else if(key=='x'){
            blur_key=0; opencv_grey_key=0;
            custom_grey_key=0; sepia_key=0;
            sobel_y_key=0; blur_quantize_key=0;
            sobel_x_key+=1;
        }
        
        // update the sobel y filter key for the video frames
        else if(key=='y'){
            blur_key=0; opencv_grey_key=0;
            custom_grey_key=0; sepia_key=0;
            sobel_x_key=0; blur_quantize_key=0;
            sobel_y_key+=1;
        }

        // update the blur quantization filter key for the video frames
        else if(key=='l'){
            blur_quantize_key=1;
            blur_key=0; opencv_grey_key=0;
            custom_grey_key=0; sepia_key=0;
            sobel_y_key=0; sobel_x_key=0;
        }

        // Quitting the program
        else if(key=='q'){
            break;
        }
    }

    delete capdev;
    return 0;
}