#include "target_detector.h"
#include <iostream>
#include <string>

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

// Function to draw a virtual cube on the image
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

// Function to draw a virtual pyramid on the image
void drawVirtualPyramid(cv::Mat &image, const cv::Mat &cameraMatrix, const cv::Mat &distCoeffs,
                        const cv::Mat &rvec, const cv::Mat &tvec, float size = 2.0f)
{
    // Define pyramid 3D coordinates
    std::vector<cv::Point3f> pyramidPoints = {
        // Base points
        {0.0f, 0.0f, 0.0f},
        {size, 0.0f, 0.0f},
        {size, -size, 0.0f},
        {0.0f, -size, 0.0f},
        // Apex point
        {size / 2, -size / 2, size * 1.5f}};

    // Project 3D points to image plane
    std::vector<cv::Point2f> imagePoints;
    cv::projectPoints(pyramidPoints, rvec, tvec, cameraMatrix, distCoeffs, imagePoints);

    // Draw pyramid edges
    cv::Scalar baseColor(255, 0, 0); // Blue color for base
    cv::Scalar edgeColor(0, 0, 255); // Red color for edges to apex
    int thickness = 2;

    // Draw base
    for (int i = 0; i < 4; i++)
    {
        cv::line(image, imagePoints[i], imagePoints[(i + 1) % 4], baseColor, thickness);
    }

    // Draw edges to apex
    for (int i = 0; i < 4; i++)
    {
        cv::line(image, imagePoints[i], imagePoints[4], edgeColor, thickness);
    }
}

int main(int argc, char **argv)
{
    // Parse command line arguments
    bool useCamera = true;
    std::string inputPath = "";

    if (argc > 1)
    {
        inputPath = argv[1];
        useCamera = false;
    }

    // Initialize video capture
    cv::VideoCapture cap;
    cv::Mat staticImage;
    bool isStaticImage = false;

    if (useCamera)
    {
        // Open default camera
        cap.open(0);
        if (!cap.isOpened())
        {
            std::cerr << "Error: Could not open camera." << std::endl;
            return -1;
        }
        std::cout << "Using live camera feed" << std::endl;
    }
    else
    {
        // Check if input is an image or video
        staticImage = cv::imread(inputPath);
        if (!staticImage.empty())
        {
            // Input is a static image
            isStaticImage = true;
            std::cout << "Using static image: " << inputPath << std::endl;
        }
        else
        {
            // Try to open as video
            cap.open(inputPath);
            if (!cap.isOpened())
            {
                std::cerr << "Error: Could not open input file: " << inputPath << std::endl;
                return -1;
            }
            std::cout << "Using video file: " << inputPath << std::endl;
        }
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
    if (!isStaticImage && cap.isOpened())
    {
        cameraMatrix.at<double>(0, 2) = cap.get(cv::CAP_PROP_FRAME_WIDTH) / 2.0;  // cx
        cameraMatrix.at<double>(1, 2) = cap.get(cv::CAP_PROP_FRAME_HEIGHT) / 2.0; // cy
    }
    else if (isStaticImage)
    {
        cameraMatrix.at<double>(0, 2) = staticImage.cols / 2.0; // cx
        cameraMatrix.at<double>(1, 2) = staticImage.rows / 2.0; // cy
    }

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

    // For static image mode, we'll use this to toggle between virtual objects
    int virtualObjectType = 0; // 0 = axes, 1 = cube, 2 = pyramid

    while (true)
    {
        cv::Mat frame;

        if (isStaticImage)
        {
            // Use the static image
            frame = staticImage.clone();
        }
        else
        {
            // Get frame from video or camera
            cap >> frame;
            if (frame.empty())
                break;
        }

        bool found = detector.detectCorners(frame, corners);

        // Store successful detections
        if (found)
        {
            lastSuccessfulFrame = frame.clone();
            lastSuccessfulCorners = corners;
            detector.drawCorners(frame, corners);

            // If calibrated, draw 3D virtual objects
            if (isCalibrated)
            {
                cv::Mat rvec, tvec;
                cv::solvePnP(point_set, corners, cameraMatrix, distCoeffs, rvec, tvec);

                // Draw different virtual objects based on mode or key press
                if (virtualObjectType == 0)
                {
                    // Draw coordinate axes (length = 3 squares)
                    cv::drawFrameAxes(frame, cameraMatrix, distCoeffs, rvec, tvec, 3.0);
                }
                else if (virtualObjectType == 1)
                {
                    // Draw virtual cube
                    drawVirtualCube(frame, cameraMatrix, distCoeffs, rvec, tvec, 3.0);
                }
                else if (virtualObjectType == 2)
                {
                    // Draw virtual pyramid
                    drawVirtualPyramid(frame, cameraMatrix, distCoeffs, rvec, tvec, 3.0);
                }
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
        char key = cv::waitKey(isStaticImage ? 0 : 1); // Wait indefinitely for static images
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
        else if (key == 'o' && isCalibrated)
        {
            // Toggle between virtual object types
            virtualObjectType = (virtualObjectType + 1) % 3;
            std::cout << "Switched to virtual object type: " << (virtualObjectType == 0 ? "Axes" : virtualObjectType == 1 ? "Cube"
                                                                                                                          : "Pyramid")
                      << std::endl;
        }

        // Display calibration status
        cv::putText(frame,
                    "Frames: " + std::to_string(corner_list.size()) +
                        (isCalibrated ? " (Calibrated)" : " (Uncalibrated)"),
                    cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0,
                    isCalibrated ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255),
                    2);

        // Display virtual object type
        if (isCalibrated)
        {
            std::string objTypeText = "Object: ";
            if (virtualObjectType == 0)
                objTypeText += "Axes";
            else if (virtualObjectType == 1)
                objTypeText += "Cube";
            else
                objTypeText += "Pyramid";

            cv::putText(frame, objTypeText, cv::Point(10, 60),
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 255), 2);
        }

        // Display the result
        cv::imshow("Chessboard Detection", frame);

        // For static images, we need to keep refreshing the window
        if (isStaticImage)
        {
            // Reset the frame to the original image for the next iteration
            staticImage = lastSuccessfulFrame.clone();
        }
    }

    // Final statistics
    std::cout << "\nFinal Statistics:" << std::endl;
    std::cout << "Total calibration frames: " << corner_list.size() << std::endl;
    if (isCalibrated)
    {
        printCalibrationParams(cameraMatrix, distCoeffs);
    }

    if (cap.isOpened())
    {
        cap.release();
    }
    cv::destroyAllWindows();
    return 0;
}