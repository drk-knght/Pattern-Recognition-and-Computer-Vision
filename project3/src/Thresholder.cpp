#include "Thresholder.h"

using namespace cv;

Mat Thresholder::apply(const Mat& input, int threshold_value) {
    Mat output = Mat::zeros(input.size(), input.type());
    
    for (int i = 0; i < input.rows; i++) {
        for (int j = 0; j < input.cols; j++) {
            output.at<uchar>(i, j) = get_threshold_value(input.at<uchar>(i, j), threshold_value);
        }
    }
    
    return output;
}

uchar Thresholder::get_threshold_value(uchar input_pixel, int threshold_value) {
    // Binary inverse: pixels above threshold become 0, below become 255
    return (input_pixel > threshold_value) ? 0 : 255;
}