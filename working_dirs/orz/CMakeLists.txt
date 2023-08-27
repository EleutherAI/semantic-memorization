cmake_minimum_required(VERSION 3.7)
project(test)
set(ARROW_BUILD_STATIC True)

find_package(Arrow CONFIG REQUIRED)
find_package(Parquet CONFIG REQUIRED)
find_package(MPI REQUIRED)

include_directories(SYSTEM ${MPI_INCLUDE_PATH})

add_executable(save_approx_hashcounts save_approx_hashcounts.cpp)
add_executable(save_true_duplicate_counts save_true_duplicate_counts.cpp)


target_link_libraries(save_approx_hashcounts PRIVATE "$<IF:$<BOOL:${ARROW_BUILD_STATIC}>,Arrow::arrow_static,Arrow::arrow_shared>")
target_link_libraries(save_approx_hashcounts PRIVATE "$<IF:$<BOOL:${ARROW_BUILD_STATIC}>,Parquet::parquet_static,Parquet::parquet_shared>")
target_link_libraries(save_true_duplicate_counts PRIVATE "$<IF:$<BOOL:${ARROW_BUILD_STATIC}>,Arrow::arrow_static,Arrow::arrow_shared>")
target_link_libraries(save_true_duplicate_counts PRIVATE "$<IF:$<BOOL:${ARROW_BUILD_STATIC}>,Parquet::parquet_static,Parquet::parquet_shared>")
target_link_libraries(save_true_duplicate_counts PRIVATE ${MPI_C_LIBRARIES})