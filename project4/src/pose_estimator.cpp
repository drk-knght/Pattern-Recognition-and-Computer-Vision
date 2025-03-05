#include <opencv2/opencv.hpp>
#include <iostream>
#include "target_detector.h"

void loadCalibrationData(cv::Mat &cameraMatrix, cv::Mat &distCoeffs,
                         const std::string &filename)
{
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if (!fs.isOpened())
    {
        throw std::runtime_error("Could not open calibration file: " + filename);
    }

    fs["camera_matrix"] >> cameraMatrix;
    fs["distortion_coefficients"] >> distCoeffs;
    fs.release();

    std::cout << "Loaded camera matrix:\n"
              << cameraMatrix << std::endl;
    std::cout << "Loaded distortion coefficients:\n"
              << distCoeffs << std::endl;
}

void printPose(const cv::Mat &rvec, const cv::Mat &tvec)
{
    // Print raw rotation and translation vectors
    std::cout << "\rRotation vector [x y z]: ["
              << std::fixed << std::setprecision(3)
              << std::setw(7) << rvec.at<double>(0) << " "
              << std::setw(7) << rvec.at<double>(1) << " "
              << std::setw(7) << rvec.at<double>(2) << "] "
              << "Translation vector (m) [x y z]: ["
              << std::setw(7) << tvec.at<double>(0) << " "
              << std::setw(7) << tvec.at<double>(1) << " "
              << std::setw(7) << tvec.at<double>(2) << "]" << std::flush;

    // Optional: If you still want Euler angles, add them on a new line
    /*
    cv::Mat rotMat;
    cv::Rodrigues(rvec, rotMat);
    cv::Vec3d eulerAngles;
    cv::Mat_<double> rotMatrix(rotMat);

    eulerAngles[0] = atan2(rotMatrix(2,1), rotMatrix(2,2)) * 180/CV_PI;
    eulerAngles[1] = atan2(-rotMatrix(2,0),
                          sqrt(rotMatrix(2,1)*rotMatrix(2,1) +
                               rotMatrix(2,2)*rotMatrix(2,2))) * 180/CV_PI;
    eulerAngles[2] = atan2(rotMatrix(1,0), rotMatrix(0,0)) * 180/CV_PI;

    std::cout << "\nEuler (deg) [Roll Pitch Yaw]: ["
              << std::setw(7) << eulerAngles[0] << " "
              << std::setw(7) << eulerAngles[1] << " "
              << std::setw(7) << eulerAngles[2] << "]" << std::flush;
    */
}

// Add this new function to draw a virtual cube
void drawVirtualCube(cv::Mat &image, const cv::Mat &cameraMatrix, const cv::Mat &distCoeffs,
                     const cv::Mat &rvec, const cv::Mat &tvec, float size = 2.0f)
{
    // Define cube 3D coordinates (centered at origin)
    std::vector<cv::Point3f> cubePoints = {
        // Bottom face
        {0.0f, 0.0f, 0.0f},
        {size, 0.0f, 0.0f},
        {size, -size, 0.0f},
        {0.0f, -size, 0.0f},
        // Top face
        {0.0f, 0.0f, size},
        {size, 0.0f, size},
        {size, -size, size},
        {0.0f, -size, size}};

    // Project 3D points to image plane
    std::vector<cv::Point2f> imagePoints;
    cv::projectPoints(cubePoints, rvec, tvec, cameraMatrix, distCoeffs, imagePoints);

    // Draw cube edges
    cv::Scalar color(0, 255, 0); // Green color
    int thickness = 2;

    // Draw bottom face
    for (int i = 0; i < 4; i++)
    {
        cv::line(image, imagePoints[i], imagePoints[(i + 1) % 4], color, thickness);
    }
    // Draw top face
    for (int i = 0; i < 4; i++)
    {
        cv::line(image, imagePoints[i + 4], imagePoints[((i + 1) % 4) + 4], color, thickness);
    }
    // Draw vertical edges
    for (int i = 0; i < 4; i++)
    {
        cv::line(image, imagePoints[i], imagePoints[i + 4], color, thickness);
    }
}

// Add this function to draw corner numbers
void drawCornerNumbers(cv::Mat &image, const std::vector<cv::Point2f> &corners)
{
    for (size_t i = 0; i < corners.size(); i++)
    {
        cv::putText(image, std::to_string(i),
                    cv::Point(corners[i].x + 10, corners[i].y + 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
    }
}

void drawVirtualHouse(cv::Mat &image, const cv::Mat &cameraMatrix, const cv::Mat &distCoeffs,
                      const cv::Mat &rvec, const cv::Mat &tvec, float size = 2.0f)
{
    // Define house points in 3D space
    std::vector<cv::Point3f> housePoints = {
        // Base points
        {0.0f, 0.0f, 0.0f},  // 0: front-left-bottom
        {size, 0.0f, 0.0f},  // 1: front-right-bottom
        {size, -size, 0.0f}, // 2: back-right-bottom
        {0.0f, -size, 0.0f}, // 3: back-left-bottom

        // Roof level points
        {0.0f, 0.0f, size},  // 4: front-left-top
        {size, 0.0f, size},  // 5: front-right-top
        {size, -size, size}, // 6: back-right-top
        {0.0f, -size, size}, // 7: back-left-top

        // Roof peak point
        {size / 2, -size / 2, size * 1.5f}, // 8: roof peak

        // Door points
        {size * 0.3f, 0.0f, 0.0f},        // 9: door-left-bottom
        {size * 0.5f, 0.0f, 0.0f},        // 10: door-right-bottom
        {size * 0.3f, 0.0f, size * 0.4f}, // 11: door-left-top
        {size * 0.5f, 0.0f, size * 0.4f}, // 12: door-right-top

        // Chimney points
        {size * 0.7f, -size * 0.3f, size},        // 13: chimney-bottom
        {size * 0.7f, -size * 0.3f, size * 1.3f}, // 14: chimney-top
    };

    // Project 3D points to image plane
    std::vector<cv::Point2f> imagePoints;
    cv::projectPoints(housePoints, rvec, tvec, cameraMatrix, distCoeffs, imagePoints);

    // Define colors
    cv::Scalar wallColor(0, 255, 0);        // Green for walls
    cv::Scalar roofColor(0, 0, 255);        // Red for roof
    cv::Scalar doorColor(255, 165, 0);      // Orange for door
    cv::Scalar chimneyColor(128, 128, 128); // Gray for chimney
    int thickness = 2;

    // Draw base walls
    for (int i = 0; i < 4; i++)
    {
        cv::line(image, imagePoints[i], imagePoints[(i + 1) % 4], wallColor, thickness);
        cv::line(image, imagePoints[i + 4], imagePoints[((i + 1) % 4) + 4], wallColor, thickness);
        cv::line(image, imagePoints[i], imagePoints[i + 4], wallColor, thickness);
    }

    // Draw roof
    for (int i = 4; i < 8; i++)
    {
        cv::line(image, imagePoints[i], imagePoints[8], roofColor, thickness);
    }

    // Draw door
    cv::line(image, imagePoints[9], imagePoints[10], doorColor, thickness);  // bottom
    cv::line(image, imagePoints[11], imagePoints[12], doorColor, thickness); // top
    cv::line(image, imagePoints[9], imagePoints[11], doorColor, thickness);  // left
    cv::line(image, imagePoints[10], imagePoints[12], doorColor, thickness); // right

    // Draw chimney
    cv::line(image, imagePoints[13], imagePoints[14], chimneyColor, thickness);
}

int main()
{
    // Load calibration data
    cv::Mat cameraMatrix, distCoeffs;
    try
    {
        loadCalibrationData(cameraMatrix, distCoeffs, "calibration_data.yml");
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        return -1;
    }

    // Initialize video capture
    cv::VideoCapture cap(0);
    if (!cap.isOpened())
    {
        std::cerr << "Error: Could not open camera." << std::endl;
        return -1;
    }

    // Initialize target detector
    TargetDetector detector(9, 6); // 9x6 internal corners

    // Create 3D points of the checkerboard in its own coordinate system
    std::vector<cv::Vec3f> objectPoints;
    for (int y = 0; y < 6; y++)
    {
        for (int x = 0; x < 9; x++)
        {
            objectPoints.push_back(cv::Vec3f(x, -y, 0.0f));
        }
    }

    bool showVirtualHouse = true;
    bool showCornerNumbers = false;

    std::cout << "\nControls:" << std::endl;
    std::cout << "  q - Quit" << std::endl;
    std::cout << "  r - Reset coordinate display" << std::endl;
    std::cout << "  n - Toggle corner numbers" << std::endl;
    std::cout << "  h - Toggle virtual house" << std::endl;
    std::cout << "\nStarting pose estimation..." << std::endl;

    while (true)
    {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty())
            break;

        std::vector<cv::Point2f> corners;
        bool found = detector.detectCorners(frame, corners);

        if (found)
        {
            // Estimate pose
            cv::Mat rvec, tvec;
            cv::solvePnP(objectPoints, corners, cameraMatrix, distCoeffs, rvec, tvec);

            // Draw the coordinate axes (make them longer: 3 units)
            cv::drawFrameAxes(frame, cameraMatrix, distCoeffs, rvec, tvec, 3.0);

            // Draw virtual house if enabled
            if (showVirtualHouse)
            {
                drawVirtualHouse(frame, cameraMatrix, distCoeffs, rvec, tvec);
            }

            // Draw corners and their numbers if enabled
            detector.drawCorners(frame, corners);
            if (showCornerNumbers)
            {
                drawCornerNumbers(frame, corners);
            }

            // Print pose information
            printPose(rvec, tvec);
        }

        // Add status text
        cv::putText(frame, "Press 'n' to toggle corner numbers",
                    cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(0, 255, 0), 2);
        cv::putText(frame, "Press 'h' to toggle virtual house",
                    cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(0, 255, 0), 2);

        // Display the frame
        cv::imshow("Pose Estimation", frame);

        // Handle keyboard input
        char key = cv::waitKey(1);
        if (key == 'q')
            break;
        if (key == 'r')
        {
            std::cout << std::endl; // Reset coordinate display position
        }
        if (key == 'n')
        {
            showCornerNumbers = !showCornerNumbers;
        }
        if (key == 'h')
        {
            showVirtualHouse = !showVirtualHouse;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}