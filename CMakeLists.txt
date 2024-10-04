cmake_minimum_required(VERSION 3.29)
project(emitter CUDA)

set(CMAKE_CUDA_STANDARD 20)

# Specify the include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_executable(emitter main.cu)

set_target_properties(emitter PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

# Set CUDA architecture for RTX 3050
set_property(TARGET emitter PROPERTY CUDA_ARCHITECTURES 86)

set(CMAKE_CUDA_FLAGS "-lineinfo")