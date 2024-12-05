cmake_minimum_required(VERSION 3.15)
project(fastfilters2 LANGUAGES CXX)

function(_target_cxx_warnings target)
    target_compile_options(${target} PRIVATE
        $<$<OR:$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:GNU>>:-Wall -Wextra -Wpedantic>
        $<$<CXX_COMPILER_ID:MSVC>:/W4 /permissive->)
endfunction()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "")
set(HWY_ENABLE_INSTALL OFF CACHE BOOL "")
set(HWY_ENABLE_TESTS OFF CACHE BOOL "")
add_subdirectory(deps/highway)

find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_internal cpp/module.cpp cpp/kernel.cpp cpp/simd.cpp)
target_include_directories(_internal PRIVATE cpp)
target_link_libraries(_internal PRIVATE hwy)
_target_cxx_warnings(_internal)

install(TARGETS _internal LIBRARY DESTINATION fastfilters2)
