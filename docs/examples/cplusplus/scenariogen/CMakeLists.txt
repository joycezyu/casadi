cmake_minimum_required(VERSION 2.8.6)

#export CMAKE_PREFIX_PATH=../casadi_install/casadi/cmake/
find_package(casadi)

add_executable(cstr_sens cstr_sens.cpp)
target_link_libraries(cstr_sens casadi)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")

add_executable(cstr_3c cstr_3c.cpp)
target_link_libraries(cstr_3c casadi)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")
