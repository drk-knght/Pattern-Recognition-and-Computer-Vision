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
    src/ssim_extractor.cpp
)

# Create matching pipeline executable
add_executable(matching_pipeline 
    src/matching_pipeline.cpp
    src/csv_util.cpp
    src/feature_utils.cpp
    src/orb_extractor.cpp
    src/lbp_extractor.cpp
    src/ssim_extractor.cpp
)

# Create q5 executable
add_executable(q5 
    src/q5.cpp
)

# Create q6 executable
add_executable(q6 
    src/q6.cpp
)

# Create q7 executable
add_executable(q7 
    src/q7.cpp
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

# Link libraries for matching pipeline
target_link_libraries(matching_pipeline 
    PRIVATE
    ${OpenCV_LIBS}
    opencv_core
    opencv_highgui
    opencv_imgproc
    opencv_videoio
    opencv_objdetect
    /opt/homebrew/Cellar/onnxruntime/1.20.1_2/lib/libonnxruntime.dylib
)

# Link libraries for q5
target_link_libraries(q5 
    PRIVATE
    ${OpenCV_LIBS}
    opencv_core
    opencv_highgui
    opencv_imgproc
    opencv_videoio
    opencv_objdetect
    /opt/homebrew/Cellar/onnxruntime/1.20.1_2/lib/libonnxruntime.dylib
)

# Link libraries for q6
target_link_libraries(q6 
    PRIVATE
    ${OpenCV_LIBS}
    opencv_core
    opencv_highgui
    opencv_imgproc
    opencv_videoio
    opencv_objdetect
    /opt/homebrew/Cellar/onnxruntime/1.20.1_2/lib/libonnxruntime.dylib
)

# Link libraries for q7
target_link_libraries(q7 
    PRIVATE
    ${OpenCV_LIBS}
    opencv_core
    opencv_highgui
    opencv_imgproc
    opencv_videoio
    opencv_objdetect
    /opt/homebrew/Cellar/onnxruntime/1.20.1_2/lib/libonnxruntime.dylib
)

# Set properties for executables
foreach(target feature_extractor image_matcher matching_pipeline q5 q6 q7)
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
