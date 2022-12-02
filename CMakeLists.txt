cmake_minimum_required(VERSION 3.19...3.25)
project(fastfilters2 LANGUAGES CXX)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(FASTFILTERS2_BENCHMARK "Build benchmarks")

set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "")
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "")
set(BUILD_TESTING OFF CACHE BOOL "")
set(HWY_ENABLE_CONTRIB OFF CACHE BOOL "")
set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "")
set(HWY_ENABLE_INSTALL OFF CACHE BOOL "")

if(${FASTFILTERS2_BENCHMARK})
    add_subdirectory(deps/benchmark)
endif()
add_subdirectory(deps/highway)
add_subdirectory(deps/pybind11)

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
