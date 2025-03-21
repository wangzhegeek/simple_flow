cmake_minimum_required(VERSION 3.10)
project(SimpleFlow VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -march=native -pthread")

# 查找BLAS库
find_package(BLAS REQUIRED)
include_directories(${BLAS_INCLUDE_DIRS})

# 包含头文件目录
include_directories(${PROJECT_SOURCE_DIR}/include)

# 添加子目录
add_subdirectory(src)
add_subdirectory(examples)
add_subdirectory(tests) 