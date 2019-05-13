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

add_executable(paper_example_sens_v2 paper_example_sens_v2.cpp)
target_link_libraries(paper_example_sens_v2 casadi)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")

add_executable(paper_example_sens_v3 paper_example_sens_v3.cpp)
target_link_libraries(paper_example_sens_v3 casadi)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")

add_executable(paper_example_sens_v4 paper_example_sens_v4.cpp)
target_link_libraries(paper_example_sens_v4 casadi)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")

add_executable(paper_example_sens_v4clean paper_example_sens_v4clean.cpp)
target_link_libraries(paper_example_sens_v4clean casadi)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")

add_executable(paper_example_module paper_example_module.cpp)
target_link_libraries(paper_example_module casadi)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")


add_executable(parametric_nlp parametric_nlp.cpp)
target_link_libraries(parametric_nlp casadi)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")


add_executable(rosenbrock rosenbrock.cpp)
target_link_libraries(rosenbrock casadi)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")



add_executable(collocation_example collocation_example.cpp)
target_link_libraries(collocation_example casadi)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")


add_executable(race_car race_car.cpp)
target_link_libraries(race_car casadi)

add_executable(direct_collocation direct_collocation.cpp)
target_link_libraries(direct_collocation casadi)

add_executable(cstr_v1 cstr_v1.cpp)
target_link_libraries(cstr_v1 casadi)

add_executable(cstr_v2 cstr_v2.cpp)
target_link_libraries(cstr_v2 casadi)

add_executable(cstr_sens cstr_sens.cpp)
target_link_libraries(cstr_sens casadi)

add_executable(cstr_sens_dummy cstr_sens_dummy.cpp)
target_link_libraries(cstr_sens_dummy casadi)

add_executable(ipopt_multiplier_example ipopt_multiplier_example.cpp)
target_link_libraries(ipopt_multiplier_example casadi)