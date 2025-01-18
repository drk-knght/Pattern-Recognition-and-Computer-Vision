cmake_minimum_required(VERSION 3.10)
project(MyProject)

# Find required packages
find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(ONNXRUNTIME REQUIRED libonnxruntime)

# Include directories
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_INCLUDE_DIRS}
    /opt/homebrew/include
    /opt/homebrew/Cellar/onnxruntime/1.20.1_2/include
    /opt/homebrew/Cellar/onnxruntime/1.20.1_2/include/onnxruntime
    /opt/homebrew/Cellar/onnxruntime/1.20.1_2/include/onnxruntime/core/session
)

# Add source files
set(MY_SOURCE_FILES 
    src/vidDisplay.cpp
    src/filter.cpp
    src/faceDetect.cpp
    src/DA2Network.hpp)

# Create executable (removed .cpp extension)
add_executable(vidDisplay ${MY_SOURCE_FILES})

# Set link directories
link_directories(
    /opt/homebrew/Cellar/onnxruntime/1.20.1_2/lib
    /opt/homebrew/lib
)

# Link libraries with full path to ONNX Runtime
target_link_libraries(vidDisplay 
    PRIVATE
    ${OpenCV_LIBS}
    opencv_core
    opencv_highgui
    opencv_imgproc
    opencv_videoio
    opencv_objdetect
    /opt/homebrew/Cellar/onnxruntime/1.20.1_2/lib/libonnxruntime.dylib
)

# Add compile definitions
target_compile_definitions(vidDisplay 
    PRIVATE
    ENABLE_PRECOMPILED_HEADERS=OFF
)

# Set C++ standard
set_target_properties(vidDisplay 
    PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON
)

# Print debug information
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
message(STATUS "ONNX Runtime libraries: ${ONNXRUNTIME_LIBRARIES}")
message(STATUS "ONNX Runtime include dirs: ${ONNXRUNTIME_INCLUDE_DIRS}")
