cmake_minimum_required(VERSION 3.19...3.25)
project(fastfilters2 LANGUAGES CXX)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(FASTFILTERS2_BENCHMARK "Build benchmarks")

include(FetchContent)
FetchContent_Declare(
    highway
    GIT_REPOSITORY https://github.com/google/highway
    GIT_TAG 1.0.2
    GIT_SHALLOW ON)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11
    GIT_TAG v2.10.1
    GIT_SHALLOW ON)
FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark
    GIT_TAG v1.7.1
    GIT_SHALLOW ON)

set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "")
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "")
set(BUILD_TESTING OFF CACHE BOOL "")
set(HWY_ENABLE_CONTRIB OFF CACHE BOOL "")
set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "")
set(HWY_ENABLE_INSTALL OFF CACHE BOOL "")

set(FASTFILTERS2_DEPENDENCIES highway pybind11)
if(${FASTFILTERS2_BENCHMARK})
    list(APPEND FASTFILTERS2_DEPENDENCIES benchmark)
endif()
FetchContent_MakeAvailable(${FASTFILTERS2_DEPENDENCIES})

add_library(fastfilters2 cpp/fastfilters2.cpp)
target_link_libraries(fastfilters2 PRIVATE hwy)
target_include_directories(fastfilters2 PUBLIC cpp)

pybind11_add_module(_core cpp/_core.cpp)
target_link_libraries(_core PRIVATE fastfilters2)
install(TARGETS _core DESTINATION fastfilters2)

if(FASTFILTERS2_BENCHMARK)
    add_executable(bm cpp/benchmark.cpp)
    target_link_libraries(bm PRIVATE fastfilters2 benchmark::benchmark)
    add_custom_target(
        runbm
        "$<TARGET_FILE:bm>"
        DEPENDS bm
        WORKING_DIRECTORY ${bm_BINARY_DIR}
        VERBATIM
        USES_TERMINAL
        COMMAND_EXPAND_LISTS
    )
endif()
