cmake_minimum_required(VERSION 3.10)
project(CameraCalibration)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add executable
add_executable(calibration 
    src/main.cpp
    src/target_detector.cpp
)

# Add pose estimation executable
add_executable(pose_estimator
    src/pose_estimator.cpp
    src/target_detector.cpp
)

# Add feature detector executable
add_executable(feature_detector
    src/feature_detector.cpp
)

# Include directories for all executables
target_include_directories(calibration PRIVATE 
    ${OpenCV_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include
)

target_include_directories(pose_estimator PRIVATE 
    ${OpenCV_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include
)

target_include_directories(feature_detector PRIVATE 
    ${OpenCV_INCLUDE_DIRS}
)

# Link libraries for all executables
target_link_libraries(calibration PRIVATE 
    ${OpenCV_LIBS}
)

target_link_libraries(pose_estimator PRIVATE 
    ${OpenCV_LIBS}
)

target_link_libraries(feature_detector PRIVATE 
    ${OpenCV_LIBS}
)