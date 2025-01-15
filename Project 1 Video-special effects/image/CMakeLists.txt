cmake_minimum_required(VERSION 3.10)
project(MyProject)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
set(MY_SOURCE_FILES 
    src/imgDisplay.cpp)
add_executable(imgDisplay ${MY_SOURCE_FILES})
target_link_libraries(imgDisplay ${OpenCV_LIBS})