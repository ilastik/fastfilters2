cmake_minimum_required(VERSION 3.18)
project(fastfilters2 LANGUAGES CXX)

# Enable compiler-specific warnings for the target.
function(_target_cxx_warnings target)
    target_compile_options(${target} PRIVATE
        $<$<OR:$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:GNU>>:-Wall -Wextra -Wpedantic>
        $<$<CXX_COMPILER_ID:MSVC>:/W4 /permissive->)
endfunction()

# Set the C++ standard for the target.
function(_target_cxx_standard target std)
    set_target_properties(${target} PROPERTIES
        CXX_STANDARD ${std}
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF)
endfunction()

# Generate `compile_commands.json` for clangd and other development tools.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Configure CMake to perform an optimized release build by default unless another build
# type is specified. Optimized builds are very important for performance-critical code.
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(FETCHCONTENT_BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/_deps)
include(FetchContent)

# Configure Highway
set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "")
set(HWY_ENABLE_TESTS OFF CACHE BOOL "")
set(HWY_ENABLE_INSTALL OFF CACHE BOOL "")
FetchContent_Declare(
    highway
    GIT_REPOSITORY https://github.com/google/highway.git
    GIT_TAG 400fbf20f2e40b984be129b88f83d4748cfc26a0
)
FetchContent_MakeAvailable(highway)

find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)

add_library(simd cpp/simd.cpp)
target_include_directories(simd PRIVATE cpp)
target_link_libraries(simd PRIVATE hwy)
_target_cxx_standard(simd 17)
_target_cxx_warnings(simd)

# Save x86 assembly listings in the build directory (GCC-compatible compilers).
# target_compile_options(simd PRIVATE -save-temps=obj -masm=intel -fverbose-asm)

nanobind_add_module(_internal cpp/module.cpp)
target_link_libraries(_internal PRIVATE hwy hwy_contrib simd)
_target_cxx_standard(_internal 17)
_target_cxx_warnings(_internal)

nanobind_add_stub(_internal_stub
    MODULE _internal
    OUTPUT _internal.pyi
    PYTHON_PATH $<TARGET_FILE_DIR:_internal>/fastfilters2
    VERBOSE
    MARKER_FILE py.typed)

install(TARGETS _internal LIBRARY DESTINATION fastfilters2)
install(FILES
    $<TARGET_FILE_DIR:_internal>/py.typed
    $<TARGET_FILE_DIR:_internal>/_internal.pyi
    DESTINATION fastfilters2)
