cmake_minimum_required(VERSION 3.10)
project(MyProject)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
set(MY_SOURCE_FILES 
    src/vidDisplay.cpp
    src/filter.cpp)
add_executable(vidDisplay.cpp ${MY_SOURCE_FILES})
target_link_libraries(vidDisplay.cpp ${OpenCV_LIBS})
