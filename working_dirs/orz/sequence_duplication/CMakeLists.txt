cmake_minimum_required(VERSION 3.7)
project(semantic-memorization)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

option(ENABLE_STATIC "Build static (.a) library" ON)
find_package(MPI REQUIRED)
find_package(ZLIB REQUIRED)

include_directories(${ZLIB_INCLUDE_DIRS})
include_directories(SYSTEM ${MPI_INCLUDE_PATH})

add_subdirectory(${PROJECT_SOURCE_DIR}/cnpy)
add_executable(save_true_duplicate_counts save_true_duplicate_counts.cpp)

target_link_libraries(save_true_duplicate_counts PRIVATE ${MPI_C_LIBRARIES})
target_link_libraries(save_true_duplicate_counts PRIVATE cnpy)