cmake_minimum_required(VERSION 3.16)
project(CBR)

set(CMAKE_CXX_STANDARD 14)

include_directories(src)
add_subdirectory(src)

file(GLOB SOURCES
        src/*.hpp
)

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

FIND_PACKAGE(OpenMP REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

add_executable(CBR main.cpp ${SOURCES})