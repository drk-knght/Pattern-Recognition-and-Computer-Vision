#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include<sys/time.h>
#include <iostream>
#include<cmath>
#include "filter.h"


int gaussian_blur(const cv::Mat &src, cv::Mat &dst ){
    if(src.empty() || src.type()!=CV_8UC3){
        return -1;
    }
    cv::Mat temp=cv::Mat::zeros(src.size(),CV_16SC3);
    src.copyTo(dst);

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