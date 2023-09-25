cmake_minimum_required(VERSION 3.20)
project(fastfilters2 LANGUAGES CXX)

function(set_cxx_standard target standard)
    set_target_properties(${target} PROPERTIES
        CXX_STANDARD ${standard}
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF)
endfunction()

function(enable_cxx_warnings target)
    target_compile_options(${target} PRIVATE
        $<$<CXX_COMPILER_ID:AppleClang,Clang,GNU>:-Wall -Wextra>
        $<$<CXX_COMPILER_ID:MSVC>:/W4>)
endfunction()

set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "")
set(HWY_ENABLE_TESTS OFF CACHE BOOL "")

add_subdirectory(deps/pybind11)
add_subdirectory(deps/highway)

add_library(fastfilters2 cpp/fastfilters2.cpp)
target_link_libraries(fastfilters2 PRIVATE hwy)
target_include_directories(fastfilters2 PRIVATE cpp)
set_target_properties(fastfilters2 PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_cxx_standard(fastfilters2 17)
enable_cxx_warnings(fastfilters2)

pybind11_add_module(_core cpp/module.cpp)
target_link_libraries(_core PRIVATE fastfilters2)
set_target_properties(_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/fastfilters2)
set_cxx_standard(_core 17)
enable_cxx_warnings(_core)
