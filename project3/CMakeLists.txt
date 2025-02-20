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

# Create a library for the common code used by both executables.
add_library(core
    src/ImageProcessor.cpp
    src/Thresholder.cpp
    src/filter.cpp
    src/KMeans.cpp
    src/morphology.cpp
    src/RegionAnalyzer.cpp
    src/RegionFeatures.cpp
)

# Create the main executable using main.cpp and link with the core library.
add_executable(object_recognition src/main.cpp)
target_link_libraries(object_recognition core ${OpenCV_LIBS})

# Create the classifier executable using classifier.cpp and link with the core library.
add_executable(classifier src/classifier.cpp)
target_link_libraries(classifier core ${OpenCV_LIBS})

add_executable(classifier_vid src/classifier_vid.cpp)
target_link_libraries(classifier_vid core ${OpenCV_LIBS})