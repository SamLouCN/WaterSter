cmake_minimum_required(VERSION 3.10)

project(yolo_kit LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# OpenCV
find_package(OpenCV REQUIRED)

# Threads
find_package(Threads REQUIRED)

# json support
find_package(nlohmann_json REQUIRED)

# RDK X5 BPU SDK
# 对应源码中的 #include "dnn/hb_dnn.h"
find_path(DNN_INCLUDE_DIR
    NAMES dnn/hb_dnn.h
    PATHS /usr/include /usr/local/include
)

find_library(DNN_LIBRARY
    NAMES dnn
    PATHS
        /usr/lib
        /usr/lib/aarch64-linux-gnu
        /usr/local/lib
)

if(NOT DNN_INCLUDE_DIR OR NOT DNN_LIBRARY)
    message(FATAL_ERROR "RDK X5 DNN headers or library not found")
endif()

# 生成动态库 libyolo_detect.so
add_library(yolo_detect SHARED
    yolo_detect.cpp
    yolo.hpp
)

target_include_directories(yolo_detect PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${DNN_INCLUDE_DIR}
)

target_compile_options(yolo_detect PRIVATE
    -Wall
)

target_link_libraries(yolo_detect PRIVATE
    ${OpenCV_LIBS}
    ${DNN_LIBRARY}
    Threads::Threads
    rt
    ${CMAKE_DL_LIBS}
)

# SHARED 默认启用 PIC，此处显式声明
set_target_properties(yolo_detect PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)