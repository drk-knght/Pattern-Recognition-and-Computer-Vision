cmake_minimum_required(VERSION 3.10)
project(MyProject)
find_package(OpenCV REQUIRED)
find_package(onnxruntime REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${onnxruntime_INCLUDE_DIRS})
set(MY_SOURCE_FILES
    da2-video.cpp
    DA2Network.hpp)
add_executable(da2-video ${MY_SOURCE_FILES})
target_link_libraries(da2-video ${OpenCV_LIBS} ${onnxruntime_LIBRARIES})