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

set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "")
set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "")
set(HWY_ENABLE_TESTS OFF CACHE BOOL "")
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

add_subdirectory(deps/benchmark)
add_subdirectory(deps/googletest)
add_subdirectory(deps/highway)
add_subdirectory(deps/pybind11)

add_library(fastfilters2 cpp/fastfilters2.cpp)
target_link_libraries(fastfilters2 PUBLIC hwy)
target_include_directories(fastfilters2 PUBLIC cpp)
set_cxx_std(fastfilters2 20)
enable_cxx_warnings(fastfilters2)

pybind11_add_module(pymodule cpp/pymodule.cpp)
target_link_libraries(pymodule PRIVATE fastfilters2)
set_target_properties(pymodule PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/fastfilters2
    OUTPUT_NAME _core
)
set_cxx_std(pymodule 20)
enable_cxx_warnings(pymodule)

enable_testing()
add_executable(tests cpp/tests.cpp)
target_link_libraries(tests PRIVATE fastfilters2 GTest::gtest_main)
set_cxx_std(tests 20)
enable_cxx_warnings(tests)
include(GoogleTest)
gtest_discover_tests(tests)

add_executable(benchmarks cpp/benchmarks.cpp)
target_link_libraries(benchmarks PRIVATE fastfilters2 benchmark::benchmark)
set_cxx_std(benchmarks 20)
enable_cxx_warnings(benchmarks)
add_custom_target(bench benchmarks VERBATIM USES_TERMINAL)
