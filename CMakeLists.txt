cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(FastCqt)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# GLOBAL SET
SET(LDFLAGS "-std=c++14 -O3 -ggdb -fopenmp -Wall -D_GLIBCXX_USE_CXX11_ABI=1 -fPIC")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}${LDFLAGS}")

# EXTERN LIB: FFTW
set(FFTW_DIR ${PROJECT_SOURCE_DIR}/3rd/fftw-3.3.10/build)
link_libraries(${FFTW_DIR}/lib/libfftw3f.so)
include_directories(${FFTW_DIR}/include)

# EXTERN LIB: Eigen
SET(EIGEN_INCLUDE_PATH ${PROJECT_SOURCE_DIR}/3rd/eigen-3.3.7)
include_directories(${EIGEN_INCLUDE_PATH})
# add_definitions(-DEIGEN_FFTW_DEFAULT)

include_directories(src)
set(CoreSrc src/core.cpp)        
add_library (FastCqt SHARED ${CoreSrc})
target_link_libraries(FastCqt ${CMAKE_CXX_FLAGS})