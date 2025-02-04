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

# Set link directories
link_directories(
    /opt/homebrew/Cellar/onnxruntime/1.20.1_2/lib
    /opt/homebrew/lib
)

# Create feature extractor executable
add_executable(feature_extractor 
    src/feature_extractor.cpp
    src/csv_util.cpp
    src/feature_utils.cpp
)

# Create image matcher executable
add_executable(image_matcher 
    src/image_matcher.cpp
    src/csv_util.cpp
    src/feature_utils.cpp
)

# Link libraries for feature extractor
target_link_libraries(feature_extractor 
    PRIVATE
    ${OpenCV_LIBS}
    opencv_core
    opencv_highgui
    opencv_imgproc
    opencv_videoio
    opencv_objdetect
    /opt/homebrew/Cellar/onnxruntime/1.20.1_2/lib/libonnxruntime.dylib
)

# Link libraries for image matcher
target_link_libraries(image_matcher 
    PRIVATE
    ${OpenCV_LIBS}
    opencv_core
    opencv_highgui
    opencv_imgproc
    opencv_videoio
    opencv_objdetect
    /opt/homebrew/Cellar/onnxruntime/1.20.1_2/lib/libonnxruntime.dylib
)

# Set properties for both executables
foreach(target feature_extractor image_matcher)
    target_compile_definitions(${target} 
        PRIVATE
        ENABLE_PRECOMPILED_HEADERS=OFF
    )

    set_target_properties(${target} 
        PROPERTIES
        CXX_STANDARD 14
        CXX_STANDARD_REQUIRED ON
    )
endforeach()

# Print debug information
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
message(STATUS "ONNX Runtime libraries: ${ONNXRUNTIME_LIBRARIES}")
message(STATUS "ONNX Runtime include dirs: ${ONNXRUNTIME_INCLUDE_DIRS}")
