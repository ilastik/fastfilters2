cmake_minimum_required(VERSION 3.19...3.25)
project(fastfilters2 LANGUAGES CXX)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

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

option(BUILD_TESTING OFF)
option(HWY_ENABLE_CONTRIB OFF)
option(HWY_ENABLE_EXAMPLES OFF)
option(HWY_ENABLE_INSTALL OFF)
FetchContent_MakeAvailable(highway pybind11)

add_library(fastfilters2 cpp/fastfilters2.cpp)
target_link_libraries(fastfilters2 PRIVATE hwy)
target_include_directories(fastfilters2 PUBLIC cpp)

pybind11_add_module(_core cpp/_core.cpp)
target_link_libraries(_core PRIVATE fastfilters2)
install(TARGETS _core DESTINATION fastfilters2)
