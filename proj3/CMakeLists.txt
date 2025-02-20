cmake_minimum_required(VERSION 3.10)
project(ObjectRecognition)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(object_recognition object_recognition.cpp)
add_executable(object_recognition_custom_knn object_recognition_custom_knn.cpp)
add_executable(object_recognition_ext object_recognition_ext.cpp)

target_link_libraries(object_recognition ${OpenCV_LIBS})
target_link_libraries(object_recognition_custom_knn ${OpenCV_LIBS})
target_link_libraries(object_recognition_ext ${OpenCV_LIBS})