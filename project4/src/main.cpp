#include "target_detector.h"
#include <iostream>

// Add these functions at the top of the file, before main()
void saveCalibrationData(const cv::Mat &cameraMatrix, const cv::Mat &distCoeffs,
                         const std::string &filename)
{
    cv::FileStorage fs(filename, cv::FileStorage::WRITE);
    fs << "camera_matrix" << cameraMatrix;
    fs << "distortion_coefficients" << distCoeffs;
    fs.release();
    std::cout << "Calibration data saved to " << filename << std::endl;
}

void printCalibrationParams(const cv::Mat &cameraMatrix, const cv::Mat &distCoeffs)
{
    std::cout << "\nCamera Matrix:\n"
              << cameraMatrix << std::endl;
    std::cout << "\nDistortion Coefficients:\n"
              << distCoeffs << std::endl;
}

int main()
{
    cv::VideoCapture cap(0); // Open default camera
    if (!cap.isOpened())
    {
        std::cerr << "Error: Could not open camera." << std::endl;
        return -1;
    }

    TargetDetector detector(9, 6); // 9x6 internal corners
    std::vector<cv::Point2f> corners;
    cv::Mat lastSuccessfulFrame;
    std::vector<cv::Point2f> lastSuccessfulCorners;

    // Lists for calibration
    std::vector<std::vector<cv::Vec3f>> point_list;
    std::vector<std::vector<cv::Point2f>> corner_list;

    // Initialize calibration parameters
    cv::Mat cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
    cameraMatrix.at<double>(0, 2) = cap.get(cv::CAP_PROP_FRAME_WIDTH) / 2.0;  // cx
    cameraMatrix.at<double>(1, 2) = cap.get(cv::CAP_PROP_FRAME_HEIGHT) / 2.0; // cy

    cv::Mat distCoeffs = cv::Mat::zeros(8, 1, CV_64F);
    std::vector<cv::Mat> rvecs, tvecs;
    bool isCalibrated = false;
    const int MIN_CALIB_FRAMES = 5;

    std::cout << "Initial camera matrix:" << std::endl;
    printCalibrationParams(cameraMatrix, distCoeffs);

    // Create the 3D world points (measured in checkerboard squares)
    std::vector<cv::Vec3f> point_set;
    for (int y = 0; y < 6; y++)
    {
        for (int x = 0; x < 9; x++)
        {
            point_set.push_back(cv::Vec3f(x, -y, 0.0f)); // Z-axis towards viewer
        }
    }

    while (true)
    {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty())
            break;

        bool found = detector.detectCorners(frame, corners);

        // Store successful detections
        if (found)
        {
            lastSuccessfulFrame = frame.clone();
            lastSuccessfulCorners = corners;
            detector.drawCorners(frame, corners);

            // If calibrated, draw 3D axes
            if (isCalibrated)
            {
                cv::Mat rvec, tvec;
                cv::solvePnP(point_set, corners, cameraMatrix, distCoeffs, rvec, tvec);
                // Draw coordinate axes (length = 3 squares)
                cv::drawFrameAxes(frame, cameraMatrix, distCoeffs, rvec, tvec, 3.0);
            }

            // Print number of corners and first corner coordinates
            std::cout << "Found " << corners.size() << " corners. ";
            if (!corners.empty())
            {
                std::cout << "First corner at (" << corners[0].x << ", "
                          << corners[0].y << ")" << std::endl;
            }
        }

        // Display the result
        cv::imshow("Chessboard Detection", frame);

        // Handle keyboard input
        char key = cv::waitKey(1);
        if (key == 'q')
        {
            break;
        }
        else if (key == 's' && !lastSuccessfulCorners.empty())
        {
            corner_list.push_back(lastSuccessfulCorners);
            point_list.push_back(point_set);
            std::cout << "Saved calibration frame " << corner_list.size()
                      << " with " << lastSuccessfulCorners.size() << " corners" << std::endl;

            // Auto-calibrate if we have enough frames
            if (corner_list.size() >= MIN_CALIB_FRAMES)
            {
                std::cout << "\nAuto-calibrating with " << corner_list.size() << " frames..." << std::endl;
                double rms = cv::calibrateCamera(point_list, corner_list, frame.size(),
                                                 cameraMatrix, distCoeffs, rvecs, tvecs,
                                                 cv::CALIB_FIX_ASPECT_RATIO);

                std::cout << "Re-projection error: " << rms << " pixels" << std::endl;
                printCalibrationParams(cameraMatrix, distCoeffs);
                isCalibrated = true;
            }
        }
        else if (key == 'c' && corner_list.size() >= MIN_CALIB_FRAMES)
        {
            // Manual calibration trigger
            std::cout << "\nPerforming calibration with " << corner_list.size() << " frames..." << std::endl;
            double rms = cv::calibrateCamera(point_list, corner_list, frame.size(),
                                             cameraMatrix, distCoeffs, rvecs, tvecs,
                                             cv::CALIB_FIX_ASPECT_RATIO);

            std::cout << "Re-projection error: " << rms << " pixels" << std::endl;
            printCalibrationParams(cameraMatrix, distCoeffs);
            isCalibrated = true;
        }
        else if (key == 'w' && isCalibrated)
        {
            // Save calibration data to file
            saveCalibrationData(cameraMatrix, distCoeffs, "calibration_data.yml");
        }
        else if (key == 'v' && isCalibrated)
        {
            // Visualize camera positions (optional)
            cv::Mat vis = cv::Mat::zeros(800, 800, CV_8UC3);
            // TODO: Add visualization of camera positions using rvecs and tvecs
            cv::imshow("Camera Positions", vis);
        }

        // Display calibration status
        cv::putText(frame,
                    "Frames: " + std::to_string(corner_list.size()) +
                        (isCalibrated ? " (Calibrated)" : " (Uncalibrated)"),
                    cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0,
                    isCalibrated ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255),
                    2);

        // Display the result
        cv::imshow("Chessboard Detection", frame);
    }

    // Final statistics
    std::cout << "\nFinal Statistics:" << std::endl;
    std::cout << "Total calibration frames: " << corner_list.size() << std::endl;
    if (isCalibrated)
    {
        printCalibrationParams(cameraMatrix, distCoeffs);
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}