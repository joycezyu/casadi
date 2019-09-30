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


add_executable(min_max_3c min_max_3c.cpp cstr_model.hpp)
target_link_libraries(min_max_3c casadi MYMODEL)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")

add_executable(multistage_3c multistage_3c.cpp cstr_model.hpp)
target_link_libraries(multistage_3c casadi MYMODEL)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")

add_executable(nominal_only nominal_only.cpp cstr_model.hpp)
target_link_libraries(nominal_only casadi MYMODEL)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")

add_executable(nominal_module nominal_module.cpp cstr_model.hpp)
target_link_libraries(nominal_module casadi MYMODEL)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")

add_executable(approx_multistage_3c approx_multistage_3c.cpp cstr_model.hpp)
target_link_libraries(approx_multistage_3c casadi MYMODEL)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")