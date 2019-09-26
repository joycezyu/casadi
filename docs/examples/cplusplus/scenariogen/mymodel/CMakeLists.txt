cmake_minimum_required(VERSION 2.8.6)

project(mymodel)

set(CPP_SOURCE cstr_model.cpp)
set(HEADERS cstr_model.hpp)

add_library(mymodel ${CPP_SOURCE} )



target_include_directories(
        mymodel PUBLIC)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")
#export CMAKE_PREFIX_PATH=../casadi_install/casadi/cmake/
find_package(casadi)


