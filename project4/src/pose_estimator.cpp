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

// Add this function to draw a virtual star
void drawVirtualStar(cv::Mat &image, const cv::Mat &cameraMatrix, const cv::Mat &distCoeffs,
                     const cv::Mat &rvec, const cv::Mat &tvec, float size = 2.0f)
{
    // Define star points in 3D space
    const int numPoints = 10; // 5 outer points + 5 inner points
    std::vector<cv::Point3f> starPoints;

    float innerRadius = size * 0.4f; // Inner radius of the star
    float outerRadius = size;        // Outer radius of the star

    for (int i = 0; i < numPoints; i++)
    {
        float angle = i * 2 * CV_PI / numPoints;
        float radius = (i % 2 == 0) ? outerRadius : innerRadius;

        // Calculate x, y coordinates on the horizontal plane
        float x = radius * cos(angle) + size / 2;  // Center the star
        float y = -radius * sin(angle) - size / 2; // Center and flip y (OpenCV convention)
        float z = size / 2;                        // Raise above the ground

        starPoints.push_back(cv::Point3f(x, y, z));
    }

    // Project 3D points to image plane
    std::vector<cv::Point2f> imagePoints;
    cv::projectPoints(starPoints, rvec, tvec, cameraMatrix, distCoeffs, imagePoints);

    // Draw star edges
    cv::Scalar starColor(255, 255, 0); // Yellow color
    int thickness = 2;

    // Connect the points to form the star
    for (int i = 0; i < numPoints; i++)
    {
        cv::line(image, imagePoints[i], imagePoints[(i + 1) % numPoints], starColor, thickness);
    }

    // Optional: Draw lines from center to each point for a more detailed star
    cv::Point2f centerPoint;
    cv::Point3f center3D(size / 2, -size / 2, size / 2); // Center of the star
    std::vector<cv::Point3f> centerVec = {center3D};
    std::vector<cv::Point2f> centerImagePoint;
    cv::projectPoints(centerVec, rvec, tvec, cameraMatrix, distCoeffs, centerImagePoint);

    for (int i = 0; i < numPoints; i += 2)
    { // Connect only to outer points
        cv::line(image, centerImagePoint[0], imagePoints[i], starColor, thickness);
    }
}

// Add this function to draw a virtual wireframe torus
void drawVirtualTorus(cv::Mat &image, const cv::Mat &cameraMatrix, const cv::Mat &distCoeffs,
                      const cv::Mat &rvec, const cv::Mat &tvec, float size = 2.0f)
{
    // Torus parameters
    const float R = size * 0.6f;  // Major radius (distance from center to center of tube)
    const float r = size * 0.2f;  // Minor radius (radius of the tube)
    const int majorSegments = 16; // Number of segments around the major circle
    const int minorSegments = 8;  // Number of segments around the minor circle

    // Generate torus points
    std::vector<cv::Point3f> torusPoints;
    std::vector<std::vector<int>> torusIndices(majorSegments, std::vector<int>(minorSegments));

    int pointIndex = 0;
    for (int i = 0; i < majorSegments; i++)
    {
        float theta = i * 2 * CV_PI / majorSegments;
        float cosTheta = cos(theta);
        float sinTheta = sin(theta);

        for (int j = 0; j < minorSegments; j++)
        {
            float phi = j * 2 * CV_PI / minorSegments;
            float cosPhi = cos(phi);
            float sinPhi = sin(phi);

            // Calculate point on torus
            float x = (R + r * cosPhi) * cosTheta + size / 2;
            float y = -((R + r * cosPhi) * sinTheta + size / 2); // Flip y for OpenCV
            float z = r * sinPhi + size / 2;                     // Raise above ground

            torusPoints.push_back(cv::Point3f(x, y, z));
            torusIndices[i][j] = pointIndex++;
        }
    }

    // Project 3D points to image plane
    std::vector<cv::Point2f> imagePoints;
    cv::projectPoints(torusPoints, rvec, tvec, cameraMatrix, distCoeffs, imagePoints);

    // Draw torus edges
    cv::Scalar torusColor(255, 165, 0); // Orange color
    int thickness = 2;

    // Draw circles around the minor radius
    for (int i = 0; i < majorSegments; i++)
    {
        for (int j = 0; j < minorSegments; j++)
        {
            int nextJ = (j + 1) % minorSegments;
            cv::line(image,
                     imagePoints[torusIndices[i][j]],
                     imagePoints[torusIndices[i][nextJ]],
                     torusColor, thickness);
        }
    }

    // Draw circles around the major radius
    for (int j = 0; j < minorSegments; j++)
    {
        for (int i = 0; i < majorSegments; i++)
        {
            int nextI = (i + 1) % majorSegments;
            cv::line(image,
                     imagePoints[torusIndices[i][j]],
                     imagePoints[torusIndices[nextI][j]],
                     torusColor, thickness);
        }
    }
}

int main(int argc, char **argv)
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

    // Check if an image file was provided as a command-line argument
    bool useStaticImage = (argc > 1);
    std::string imagePath = "";
    cv::Mat staticImage;
    cv::Mat originalStaticImage; // Store the original image without any drawings

    if (useStaticImage)
    {
        imagePath = argv[1];
        staticImage = cv::imread(imagePath);
        if (staticImage.empty())
        {
            std::cerr << "Error: Could not open image file: " << imagePath << std::endl;
            return -1;
        }
        originalStaticImage = staticImage.clone(); // Keep a clean copy
        std::cout << "Using static image: " << imagePath << std::endl;
    }
    else
    {
        // Initialize video capture for live camera
        cv::VideoCapture cap(0);
        if (!cap.isOpened())
        {
            std::cerr << "Error: Could not open camera." << std::endl;
            return -1;
        }
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
    bool showVirtualCube = false;
    bool showCornerNumbers = false;
    bool showAxes = true;
    bool showVirtualStar = false;
    bool showVirtualTorus = false;

    std::cout << "\nControls:" << std::endl;
    std::cout << "  q - Quit" << std::endl;
    std::cout << "  r - Reset coordinate display" << std::endl;
    std::cout << "  n - Toggle corner numbers" << std::endl;
    std::cout << "  h - Toggle virtual house" << std::endl;
    std::cout << "  c - Toggle virtual cube" << std::endl;
    std::cout << "  a - Toggle coordinate axes" << std::endl;
    std::cout << "  s - Toggle virtual star" << std::endl;
    std::cout << "  t - Toggle virtual torus" << std::endl;
    std::cout << "\nStarting pose estimation..." << std::endl;

    cv::VideoCapture cap;
    if (!useStaticImage)
    {
        cap.open(0);
        if (!cap.isOpened())
        {
            std::cerr << "Error: Could not open camera." << std::endl;
            return -1;
        }
    }

    // Store the last detected corners and pose
    std::vector<cv::Point2f> lastCorners;
    cv::Mat lastRvec, lastTvec;
    bool lastFound = false;

    while (true)
    {
        cv::Mat frame;

        if (useStaticImage)
        {
            // Start with a clean copy of the original image
            frame = originalStaticImage.clone();
        }
        else
        {
            // Get frame from camera
            cap >> frame;
            if (frame.empty())
                break;
        }

        std::vector<cv::Point2f> corners;
        bool found = detector.detectCorners(frame, corners);

        // If corners are found, update the last successful detection
        if (found)
        {
            lastCorners = corners;
            cv::solvePnP(objectPoints, corners, cameraMatrix, distCoeffs, lastRvec, lastTvec);
            lastFound = true;

            // Always draw the detected corners
            detector.drawCorners(frame, corners);
        }

        // If we have a valid pose (either from this frame or previous), draw virtual objects
        if (lastFound)
        {
            // Draw coordinate axes if enabled
            if (showAxes)
            {
                cv::drawFrameAxes(frame, cameraMatrix, distCoeffs, lastRvec, lastTvec, 3.0);
            }

            // Draw virtual house if enabled
            if (showVirtualHouse)
            {
                drawVirtualHouse(frame, cameraMatrix, distCoeffs, lastRvec, lastTvec, 3.0);
            }

            // Draw virtual cube if enabled
            if (showVirtualCube)
            {
                drawVirtualCube(frame, cameraMatrix, distCoeffs, lastRvec, lastTvec, 3.0);
            }

            // Draw virtual star if enabled
            if (showVirtualStar)
            {
                drawVirtualStar(frame, cameraMatrix, distCoeffs, lastRvec, lastTvec, 3.0);
            }

            // Draw virtual torus if enabled
            if (showVirtualTorus)
            {
                drawVirtualTorus(frame, cameraMatrix, distCoeffs, lastRvec, lastTvec, 3.0);
            }

            // Draw corner numbers if enabled
            if (showCornerNumbers)
            {
                drawCornerNumbers(frame, lastCorners);
            }

            // Display pose information
            printPose(lastRvec, lastTvec);
        }

        // Add on-screen controls display
        int lineHeight = 20;
        int startY = 30;
        int startX = 10;
        double fontSize = 0.5;
        int thickness = 1;
        cv::Scalar textColor(0, 255, 255); // Yellow text
        cv::Scalar statusColor(0, 255, 0); // Green for ON status
        cv::Scalar bgColor(0, 0, 0, 150);  // Semi-transparent black background

        // Create a semi-transparent overlay for the controls
        cv::Mat overlay;
        frame.copyTo(overlay);
        cv::rectangle(overlay, cv::Point(5, 5), cv::Point(250, startY + 9 * lineHeight), bgColor, -1);
        cv::addWeighted(overlay, 0.7, frame, 0.3, 0, frame);

        // Draw the controls text
        cv::putText(frame, "CONTROLS:", cv::Point(startX, startY),
                    cv::FONT_HERSHEY_SIMPLEX, fontSize, textColor, thickness);

        cv::putText(frame, "q - Quit", cv::Point(startX, startY + lineHeight),
                    cv::FONT_HERSHEY_SIMPLEX, fontSize, textColor, thickness);

        cv::putText(frame, "r - Reset display", cv::Point(startX, startY + 2 * lineHeight),
                    cv::FONT_HERSHEY_SIMPLEX, fontSize, textColor, thickness);

        // Show toggle states with ON/OFF indicators
        std::string cornerStatus = showCornerNumbers ? "ON" : "OFF";
        cv::putText(frame, "n - Corner numbers: " + cornerStatus,
                    cv::Point(startX, startY + 3 * lineHeight),
                    cv::FONT_HERSHEY_SIMPLEX, fontSize, textColor, thickness);

        std::string houseStatus = showVirtualHouse ? "ON" : "OFF";
        cv::putText(frame, "h - Virtual house: " + houseStatus,
                    cv::Point(startX, startY + 4 * lineHeight),
                    cv::FONT_HERSHEY_SIMPLEX, fontSize, textColor, thickness);

        std::string cubeStatus = showVirtualCube ? "ON" : "OFF";
        cv::putText(frame, "c - Virtual cube: " + cubeStatus,
                    cv::Point(startX, startY + 5 * lineHeight),
                    cv::FONT_HERSHEY_SIMPLEX, fontSize, textColor, thickness);

        std::string starStatus = showVirtualStar ? "ON" : "OFF";
        cv::putText(frame, "s - Virtual star: " + starStatus,
                    cv::Point(startX, startY + 6 * lineHeight),
                    cv::FONT_HERSHEY_SIMPLEX, fontSize, textColor, thickness);

        std::string torusStatus = showVirtualTorus ? "ON" : "OFF";
        cv::putText(frame, "t - Virtual torus: " + torusStatus,
                    cv::Point(startX, startY + 7 * lineHeight),
                    cv::FONT_HERSHEY_SIMPLEX, fontSize, textColor, thickness);

        std::string axesStatus = showAxes ? "ON" : "OFF";
        cv::putText(frame, "a - Coordinate axes: " + axesStatus,
                    cv::Point(startX, startY + 8 * lineHeight),
                    cv::FONT_HERSHEY_SIMPLEX, fontSize, textColor, thickness);

        // Display the result
        cv::imshow("Pose Estimation", frame);

        // Handle keyboard input with different wait times for static vs. live
        char key = cv::waitKey(useStaticImage ? 10 : 1); // Short wait for static images to allow UI updates

        if (key == 'q')
        {
            break;
        }
        else if (key == 'r')
        {
            // Reset coordinate display
            lastFound = false;
        }
        else if (key == 'n')
        {
            showCornerNumbers = !showCornerNumbers;
            std::cout << "\nCorner numbers: " << (showCornerNumbers ? "ON" : "OFF") << std::endl;
        }
        else if (key == 'h')
        {
            showVirtualHouse = !showVirtualHouse;
            std::cout << "\nVirtual house: " << (showVirtualHouse ? "ON" : "OFF") << std::endl;
        }
        else if (key == 'c')
        {
            showVirtualCube = !showVirtualCube;
            std::cout << "\nVirtual cube: " << (showVirtualCube ? "ON" : "OFF") << std::endl;
        }
        else if (key == 'a')
        {
            showAxes = !showAxes;
            std::cout << "\nCoordinate axes: " << (showAxes ? "ON" : "OFF") << std::endl;
        }
        else if (key == 's')
        {
            showVirtualStar = !showVirtualStar;
            std::cout << "\nVirtual star: " << (showVirtualStar ? "ON" : "OFF") << std::endl;
        }
        else if (key == 't')
        {
            showVirtualTorus = !showVirtualTorus;
            std::cout << "\nVirtual torus: " << (showVirtualTorus ? "ON" : "OFF") << std::endl;
        }
    }

    if (!useStaticImage)
    {
        cap.release();
    }
    cv::destroyAllWindows();
    return 0;
}