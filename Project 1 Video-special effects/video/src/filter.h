/*
    Agnibha Chatterjee
    Om Agarwal
    Jan 12 2024
    CS5330- Pattern Recognition & Computer Vision
    This file is the entry 
*/


#include <opencv2/opencv.hpp>

int greyscale(cv::Mat &src, cv::Mat &dst);

int sepia(cv::Mat &src, cv::Mat &dst);

int blur5x5_1( cv::Mat &src, cv::Mat &dst );

int blur5x5_2( cv::Mat &src, cv::Mat &dst );

int sobelX3x3( cv::Mat &src, cv::Mat &dst );

int sobelY3x3( cv::Mat &src, cv::Mat &dst );

int magnitude( cv::Mat &sx, cv::Mat &sy, cv::Mat &dst );

int blurQuantize( cv::Mat &src, cv::Mat &dst, int levels );

