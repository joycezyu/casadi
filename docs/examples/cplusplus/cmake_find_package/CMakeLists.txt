cmake_minimum_required(VERSION 2.8.6)

# export CMAKE_PREFIX_PATH=../casadi_install/casadi/cmake/
find_package(casadi)

add_executable(casadi_demo casadi_demo.cpp)
target_link_libraries(casadi_demo casadi)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")


add_executable(casadi_demo2 casadi_demo2.cpp)
target_link_libraries(casadi_demo2 casadi)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")


add_executable(casadi_gjh casadi_gjh.cpp)
target_link_libraries(casadi_gjh casadi)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")

add_executable(casadi_gjh2 casadi_gjh2.cpp)
target_link_libraries(casadi_gjh2 casadi)
# target_link_libraries(casadi_gjh2 sx)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")


add_executable(paper_example_sens paper_example_sens.cpp)
target_link_libraries(paper_example_sens casadi)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")



add_executable(rosenbrock rosenbrock.cpp)
target_link_libraries(rosenbrock casadi)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")

add_executable(callback callback.cpp)
target_link_libraries(callback casadi)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")
