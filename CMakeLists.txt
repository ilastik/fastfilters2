cmake_minimum_required(VERSION 3.19)
project(fastfilters2 LANGUAGES CXX)

function(set_cxx_std target std)
    set_target_properties(${target} PROPERTIES
        CXX_STANDARD ${std}
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

add_subdirectory(deps/highway)
add_subdirectory(deps/pybind11)

add_library(fastfilters2 cpp/fastfilters2.cpp)
target_link_libraries(fastfilters2 PRIVATE hwy)
target_include_directories(fastfilters2 PRIVATE cpp)
set_cxx_std(fastfilters2 17)
enable_cxx_warnings(fastfilters2)

pybind11_add_module(pymodule cpp/pymodule.cpp)
target_link_libraries(pymodule PRIVATE fastfilters2)
set_target_properties(pymodule PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/fastfilters2
    OUTPUT_NAME _core
)
set_cxx_std(pymodule 17)
enable_cxx_warnings(pymodule)
