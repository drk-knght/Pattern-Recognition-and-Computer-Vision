#include <opencv2/opencv.hpp>
#include <iostream>

class FeatureDetector
{
private:
    // Harris corner parameters
    int blockSize = 2;
    int apertureSize = 3;
    double k = 0.04;
    double threshold = 150;

    // FAST parameters
    int fastThreshold = 20;
    bool nonmaxSuppression = true;

public:
    enum DetectorType
    {
        HARRIS,
        FAST
    };
    DetectorType currentDetector = HARRIS;

    void detectAndDrawFeatures(cv::Mat &frame)
    {
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        if (currentDetector == HARRIS)
        {
            detectHarrisCorners(gray, frame);
        }
        else
        {
            detectFastFeatures(gray, frame);
        }

        // Draw current parameters
        std::string detectorName = (currentDetector == HARRIS) ? "Harris" : "FAST";
        cv::putText(frame, "Detector: " + detectorName,
                    cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    cv::Scalar(0, 255, 0), 2);

        if (currentDetector == HARRIS)
        {
            cv::putText(frame, "Threshold: " + std::to_string(int(threshold)),
                        cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                        cv::Scalar(0, 255, 0), 2);
            cv::putText(frame, "Block Size: " + std::to_string(blockSize),
                        cv::Point(10, 90), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                        cv::Scalar(0, 255, 0), 2);
        }
        else
        {
            cv::putText(frame, "Threshold: " + std::to_string(fastThreshold),
                        cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                        cv::Scalar(0, 255, 0), 2);
            cv::putText(frame, "NonMaxSuppression: " + std::to_string(nonmaxSuppression),
                        cv::Point(10, 90), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                        cv::Scalar(0, 255, 0), 2);
        }
    }

    void detectHarrisCorners(const cv::Mat &gray, cv::Mat &output)
    {
        cv::Mat dst = cv::Mat::zeros(gray.size(), CV_32FC1);

        // Detect Harris corners
        cv::cornerHarris(gray, dst, blockSize, apertureSize, k);

        // Normalize and scale
        cv::Mat dst_norm;
        cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());

        // Draw corners
        for (int i = 0; i < dst_norm.rows; i++)
        {
            for (int j = 0; j < dst_norm.cols; j++)
            {
                if ((int)dst_norm.at<float>(i, j) > threshold)
                {
                    cv::circle(output, cv::Point(j, i), 5, cv::Scalar(0, 0, 255), 2);
                }
            }
        }
    }

    void detectFastFeatures(const cv::Mat &gray, cv::Mat &output)
    {
        std::vector<cv::KeyPoint> keypoints;
        cv::FAST(gray, keypoints, fastThreshold, nonmaxSuppression);

        // Draw detected features
        for (const auto &kp : keypoints)
        {
            cv::circle(output, kp.pt, 5, cv::Scalar(255, 0, 0), 2);
        }
    }

    void adjustParameters(char key)
    {
        switch (key)
        {
        case 'd': // Switch detector
            currentDetector = (currentDetector == HARRIS) ? FAST : HARRIS;
            break;

        // Harris parameters
        case 'q':
            threshold = std::max(0.0, threshold - 10);
            break;
        case 'w':
            threshold += 10;
            break;
        case 'a':
            blockSize = std::max(2, blockSize - 1);
            break;
        case 's':
            blockSize++;
            break;

        // FAST parameters
        case 'z':
            fastThreshold = std::max(1, fastThreshold - 1);
            break;
        case 'x':
            fastThreshold++;
            break;
        case 'n':
            nonmaxSuppression = !nonmaxSuppression;
            break;
        }
    }
};

int main()
{
    cv::VideoCapture cap(0);
    if (!cap.isOpened())
    {
        std::cerr << "Error: Could not open camera." << std::endl;
        return -1;
    }

    FeatureDetector detector;

    std::cout << "\nControls:" << std::endl;
    std::cout << "  q/w - Decrease/Increase threshold" << std::endl;
    std::cout << "  a/s - Decrease/Increase block size (Harris)" << std::endl;
    std::cout << "  z/x - Decrease/Increase FAST threshold" << std::endl;
    std::cout << "  n   - Toggle non-max suppression (FAST)" << std::endl;
    std::cout << "  d   - Switch between Harris and FAST" << std::endl;
    std::cout << "  ESC - Quit" << std::endl;

    while (true)
    {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty())
            break;

        detector.detectAndDrawFeatures(frame);

        cv::imshow("Feature Detection", frame);

        char key = cv::waitKey(1);
        if (key == 27)
            break; // ESC key
        detector.adjustParameters(key);
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}