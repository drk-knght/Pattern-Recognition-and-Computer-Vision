/*
    Agnibha Chatterjee
    Om Agarwal
    Jan 12 2024
    CS5330- Pattern Recognition & Computer Vision
    This file is the entry 
*/

#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include<sys/time.h>
#include <iostream>
#include<cmath>
#include "filter.h"

// returns a double which gives time in seconds
double getTime() {
  struct timeval cur;

  gettimeofday( &cur, NULL );
  return( cur.tv_sec + cur.tv_usec / 1000000.0 );
}
  

//customized grayscale filter by averaging the channels
int greyscale(cv::Mat &src, cv::Mat &dst){
    if(src.empty() || src.type()!=CV_8UC3){
        return -1;
    }
    if(src.empty()){
        return -1;
    }
    cv::Mat single_channel[3];
    cv::split(src,single_channel);
    
    cv::Mat b=single_channel[0];
    cv::Mat g=single_channel[1];
    cv::Mat r=single_channel[2];

    cv::Mat avg_channel_value=(b+g+r)/3;

    single_channel[0]=avg_channel_value;
    single_channel[1]=avg_channel_value;
    single_channel[2]=avg_channel_value;

    cv::merge(single_channel,3,dst);

    return 0;
}

//Sepia Filter
/*
0.272, 0.534, 0.131    // Blue coefficients for R, G, B  (pay attention to the channel order)
0.349, 0.686, 0.168    // Green coefficients
0.393, 0.769, 0.189    // Red coefficients
*/
int sepia(cv::Mat &src, cv::Mat &dst){
    if(src.empty() || src.type()!=CV_8UC3){
        return -1;
    }
    dst=cv::Mat::zeros(src.size(),CV_8UC3);

    for(int i=0;i<src.rows;i++){
        for(int j=0;j<src.cols;j++){
            
            int src_blue_pixel=src.at<cv::Vec3b>(i, j)[0];
            int src_green_pixel=src.at<cv::Vec3b>(i, j)[1];
            int src_red_pixel=src.at<cv::Vec3b>(i, j)[2];

            float blue_dst = 0.131f * src_blue_pixel + 0.534f * src_green_pixel + 0.272f * src_red_pixel;
            float green_dst = (0.168f * src_blue_pixel + 0.686f * src_green_pixel + 0.349f * src_red_pixel)/1.2;
            float red_dst = (0.189f * src_blue_pixel + 0.769f * src_green_pixel + 0.393f * src_red_pixel)/1.35;
            
            dst.at<cv::Vec3b>(i, j)[0]=(unsigned char)(blue_dst);
            dst.at<cv::Vec3b>(i, j)[1]=(unsigned char)(green_dst);
            dst.at<cv::Vec3b>(i, j)[2]=(unsigned char)(red_dst);
        }
    }
    return 0;
}

// Gaussian Filter
/*
1 2 4 2 1; 
2 4 8 4 2; 
4 8 16 8 4; 
2 4 8 4 2; 
1 2 4 2 1;
*/
int blur5x5_1( cv::Mat &src, cv::Mat &dst ){
    if(src.empty() || src.type()!=CV_8UC3){
        return -1;
    }
    // set up the timing for version 1
    double startTime = getTime();
    src.copyTo(dst);
    int kernel[5][5]={
        {1, 2, 4, 2, 1},
        {2, 4, 8, 4, 2},
        {4, 8, 16, 8, 4},
        {2, 4, 8, 4, 2},
        {1, 2, 4, 2, 1}};
    
    int normalize_value=100; // sum of the coefficients for gaussian filter
    
    for(int i=2;i<src.rows-2;i++){
        for(int j=2;j<src.cols-2;j++){
            int blue=0,green=0,red=0;

            for(int dx=-2;dx<=2;dx++){
                for(int dy=-2;dy<=2;dy++){
                    cv::Vec3b pixel=src.at<cv::Vec3b>(i+dx,j+dy);
                    int kernel_weight=kernel[dx+2][dy+2];
                    
                    blue+=kernel_weight*pixel[0];
                    green+=kernel_weight*pixel[1];
                    red+=kernel_weight*pixel[2];
                }
            }
            dst.at<cv::Vec3b>(i,j)[0]=blue/normalize_value;
            dst.at<cv::Vec3b>(i,j)[1]=green/normalize_value;
            dst.at<cv::Vec3b>(i,j)[2]=red/normalize_value;
        }
    }
    // end time for version 1
    double endTime=getTime();
    // compute the time per image
    double difference = (endTime - startTime);
    // print the results
    printf("Time per image (1): %.4lf seconds\n", difference );
    return 0;
}

// 1 2 4 2 1
int blur5x5_2( cv::Mat &src, cv::Mat &dst ){
    if(src.empty() || src.type()!=CV_8UC3){
        return -1;
    }
    cv::Mat temp=cv::Mat::zeros(src.size(),CV_16SC3);
    src.copyTo(dst);
    // dst.convertTo(dst,CV_16SC3);

    for(int i=2;i+2<src.rows;i++){
        for(int j=2;j+2<src.cols;j++){
            for(int channel=0;channel<3;channel++){
                temp.at<cv::Vec3s>(i,j)[channel]=1*src.at<cv::Vec3b>(i,j-2)[channel]+ 2*src.at<cv::Vec3b>(i,j-1)[channel]+
                                                 1*src.at<cv::Vec3b>(i,j+2)[channel]+ 2*src.at<cv::Vec3b>(i,j+1)[channel]+
                                                 4*src.at<cv::Vec3b>(i,j)[channel];
            }
        }
    }

    for(int i=2;i+2<src.rows;i++){
        for(int j=2;j+2<src.cols;j++){
            for(int channel=0;channel<3;channel++){
                int blur_value=1*temp.at<cv::Vec3s>(i-2,j)[channel]+ 2*temp.at<cv::Vec3s>(i-1,j)[channel]+
                                1*temp.at<cv::Vec3s>(i+2,j)[channel]+ 2*temp.at<cv::Vec3s>(i+1,j)[channel]+
                                4*temp.at<cv::Vec3s>(i,j)[channel];
                blur_value=blur_value/100;
                dst.at<cv::Vec3b>(i,j)[channel]=blur_value;
            }
        }
    }
    return 0;
}

// Positive Right Sobel X Filter
/*
    -1 0 1
    -2 0 2
    -1 0 1
*/
int sobelX3x3( cv::Mat &src, cv::Mat &dst ){
    if(src.empty() || src.type()!=CV_8UC3){
        return -1;
    }
    dst=cv::Mat::zeros(src.size(),CV_16SC3);
    cv::Mat temp=cv::Mat::zeros(src.size(),CV_16SC3);
    for(int i=1;i+1<src.rows;i++){
        for(int j=1;j+1<src.cols;j++){
            for(int channel=0;channel<3;channel++){
                temp.at<cv::Vec3s>(i,j)[channel]=-1*src.at<cv::Vec3b>(i,j-1)[channel]
                                                +0*src.at<cv::Vec3b>(i,j)[channel]
                                                 +1*src.at<cv::Vec3b>(i,j+1)[channel];
            }
        }
    }
    int mx=0;
    for(int i=1;i+1<src.rows;i++){
        for(int j=1;j+1<src.cols;j++){
            for(int channel=0;channel<3;channel++){
                int temp_sobel_x_value=1*temp.at<cv::Vec3s>(i-1,j)[channel]
                                        +2*temp.at<cv::Vec3s>(i,j)[channel]
                                        +1*temp.at<cv::Vec3s>(i+1,j)[channel];
                temp_sobel_x_value=temp_sobel_x_value/4;
                dst.at<cv::Vec3s>(i,j)[channel]=temp_sobel_x_value;
                if(temp_sobel_x_value>mx){
                    mx=temp_sobel_x_value;
                }
            }
        }
    }
    return 0;
}

// Positive UP Sobel Y Filter
/*
    1 2 1
    0 0 0
    -1 -2 -1
*/
int sobelY3x3( cv::Mat &src, cv::Mat &dst ){
    if(src.empty() || src.type()!=CV_8UC3){
        return -1;
    }
    dst=cv::Mat::zeros(src.size(),CV_16SC3);
    cv::Mat temp=cv::Mat::zeros(src.size(),CV_16SC3);
    for(int i=1;i+1<src.rows;i++){
        for(int j=1;j+1<src.cols;j++){
            for(int channel=0;channel<3;channel++){
                temp.at<cv::Vec3s>(i,j)[channel]=-1*src.at<cv::Vec3b>(i-1,j)[channel]+1*src.at<cv::Vec3b>(i+1,j)[channel];
            }
        }
    }
    for(int i=1;i+1<src.rows;i++){
        for(int j=1;j+1<src.cols;j++){
            for(int channel=0;channel<3;channel++){
                int temp_sobel_x_value=1*temp.at<cv::Vec3s>(i,j-1)[channel]+2*temp.at<cv::Vec3s>(i,j)[channel]+1*temp.at<cv::Vec3s>(i,j+1)[channel];
                temp_sobel_x_value=temp_sobel_x_value/4;
                dst.at<cv::Vec3s>(i,j)[channel]=temp_sobel_x_value;
            }
        }
    }
    return 0;
}

int magnitude( cv::Mat &sx, cv::Mat &sy, cv::Mat &dst ){
    if(sx.empty() || sy.empty()){
        return -1;
    }
    if(sx.type()!=CV_16SC3 || sy.type()!=CV_16SC3){
        return -1;
    }
    dst=cv::Mat::zeros(sx.size(),CV_8UC3);

    for(int i=0;i<sx.rows;i++){
        for(int j=0;j<sx.cols;j++){
            for(int channel=0;channel<3;channel++){
                float sx_val=sx.at<cv::Vec3s>(i,j)[channel];
                float sy_val=sy.at<cv::Vec3s>(i,j)[channel];
                float temp=sqrt(sx_val*sx_val + sy_val*sy_val);
                dst.at<cv::Vec3b>(i,j)[channel]=temp/1.4f;
            }
        }
    }
    return 0;
}

int blurQuantize( cv::Mat &src, cv::Mat &dst, int levels ){
    if(src.empty() || src.type()!=CV_8UC3){
        return -1;
    }
    // src.copyTo(dst);
    cv::Mat temp=cv::Mat::zeros(src.size(),CV_8UC3);
    int res=blur5x5_2(src,temp);

    if(res==-1){
        return -1;
    }
    int b=255/levels;
    for(int i=0;i<src.rows;i++){
        for(int j=0;j<src.cols;j++){

            for(int channel=0;channel<3;channel++){
                int xt=src.at<cv::Vec3b>(i,j)[channel];
                xt=xt/b;
                int xf=xt*b;
                dst.at<cv::Vec3b>(i,j)[channel]=xf;
            }
        }
    }
    return 0;
}

int isolateRed(cv:: Mat &src, cv::Mat dst){
    if(src.empty() || src.type()!=CV_8UC3){
        return -1;
    }
    cv::Mat hsv, mask, grey;
    cv::cvtColor(dst, hsv, cv::COLOR_BGR2HSV);
    cv::inRange(hsv, cv::Scalar(160, 100, 100), cv::Scalar(180, 255, 255), mask);

    cv::cvtColor(dst, grey, cv::COLOR_BGR2GRAY);
    cv::cvtColor(grey, dst, cv::COLOR_GRAY2BGR);

    cv::Mat colored;
    src.copyTo(colored, mask);
    cv::add(dst, colored, dst);
    
    return 0;
}
