cmake_minimum_required(VERSION 3.10)
project(object_recognition)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)

# Add source files
set(SOURCES
    src/main.cpp
    src/ImageProcessor.cpp
    src/Thresholder.cpp
    src/filter.cpp
    src/KMeans.cpp
    src/morphology.cpp
    src/RegionAnalyzer.cpp
    src/RegionFeatures.cpp
)

# Create executable
add_executable(object_recognition ${SOURCES})

# Link libraries
target_link_libraries(object_recognition ${OpenCV_LIBS})