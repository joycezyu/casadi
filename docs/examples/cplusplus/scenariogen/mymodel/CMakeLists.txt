cmake_minimum_required(VERSION 2.8.6)

set(CPP_SOURCE cstr_model.cpp)
set(HEADERS cstr_model.hpp)



add_library(MYMODEL ${CPP_SOURCE} )

find_library(MYMODEL mymodel ../../cplusplus/scenariogen/mymodel)
add_executable(cstr controller_cstr_3c_nmpc_modularize.cpp cstr_model.hpp)
target_link_libraries(cstr casadi MYMODEL)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")
#export CMAKE_PREFIX_PATH=../casadi_install/casadi/cmake/
find_package(casadi)


