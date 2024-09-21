cmake_minimum_required(VERSION 3.29)
project(emitter CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(emitter main.cu)

set_target_properties(emitter PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
