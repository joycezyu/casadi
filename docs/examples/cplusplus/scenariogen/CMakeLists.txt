cmake_minimum_required(VERSION 2.8.6)

#export CMAKE_PREFIX_PATH=../casadi_install/casadi/cmake/
find_package(casadi)

#add_executable(cstr_sens cstr_sens.cpp)
#target_link_libraries(cstr_sens casadi)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")

#add_executable(cstr_sens_3c cstr_sens_3c.cpp)
#target_link_libraries(cstr_sens_3c casadi)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")

add_executable(cstr_sens_3c_v1 cstr_sens_3c_v1.cpp)
target_link_libraries(cstr_sens_3c_v1 casadi)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")

add_executable(cstr_sens_3c_v1_dg cstr_sens_3c_v1_dg.cpp)
target_link_libraries(cstr_sens_3c_v1_dg casadi)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")

add_executable(cstr_sens_3c_whole cstr_sens_3c_whole.cpp)
target_link_libraries(cstr_sens_3c_whole casadi)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")


#add_executable(cstr_3c cstr_3c.cpp)
#target_link_libraries(cstr_3c casadi)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")

#add_executable(cstr_3c_v1 cstr_3c_v1.cpp)
#target_link_libraries(cstr_3c_v1 casadi)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")

add_executable(cstr_3c_v2 cstr_3c_v2.cpp)
target_link_libraries(cstr_3c_v2 casadi)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")