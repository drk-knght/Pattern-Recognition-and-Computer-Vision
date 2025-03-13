cmake_minimum_required(VERSION 3.10)
project(CameraCalibration)

find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(calibration 
    src/main.cpp
    src/target_detector.cpp
)

add_executable(pose_estimator
    src/pose_estimator.cpp
    src/target_detector.cpp
)

add_executable(feature_detector
    src/feature_detector.cpp
)

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

target_link_libraries(calibration PRIVATE 
    ${OpenCV_LIBS}
)

target_link_libraries(pose_estimator PRIVATE 
    ${OpenCV_LIBS}
)

target_link_libraries(feature_detector PRIVATE 
    ${OpenCV_LIBS}
)