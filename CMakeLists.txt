cmake_minimum_required(VERSION 3.30)
project(OpencvTest)
set(CMAKE_CXX_STANDARD 11)

# 查找 OpenCV 库
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# add CUDA
set(CUDA_LIBRARIES_DIRS "/usr/local/cuda-11.8/targets/x86_64-linux/lib")
set(CUDA_INCLUDE_DIRS "/usr/local/cuda-11.8/targets/x86_64-linux/include")
include_directories(${CUDA_INCLUDE_DIRS})
file(GLOB CUDA_LIBRARIES "${CUDA_LIBRARIES_DIRS}/*.so" "${CUDA_LIBRARIES_DIRS}/*.a")

#tensorrt
include_directories(/home/wyh/TensorRT-10.8.0.43/include)
set(TENSORRT_LIB_PATH "/home/wyh/TensorRT-10.8.0.43/lib")
file(GLOB LIBS "${TENSORRT_LIB_PATH}/*.so")


add_executable(OpencvTest main.cpp
)

target_link_libraries(OpencvTest ${LIBS} ${CUDA_LIBRARIES} ${OpenCV_LIBS})