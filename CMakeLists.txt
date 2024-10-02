cmake_minimum_required(VERSION 3.5.0)
project(espressoSq VERSION 0.1.0 LANGUAGES C CXX)

set(SOURCE_FILES_COMMON src/sq_avx.cpp)
set(SOURCE_FILES_AVX2 "")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -fPIC")

include(CheckCXXCompilerFlag)

# Check for AVX or AVX2 support first. Rn library only works with x86 intrinsics, but i plan to add arm support with NEON in the future.
check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i[3-6]86")
    message(STATUS "CPU architecture is x86")
    add_definitions(-DARCH_X86)
    if(COMPILER_SUPPORTS_AVX2)
        message(STATUS "Compiler supports AVX2")
        add_definitions(-DSUPPORTS_AVX2)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
        set(SOURCE_FILES_AVX2 src/approx_trig_avx2.cpp)
    else()
        message(STATUS "Compiler does not support AVX2")
    endif()
endif()


if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64|ARM")
    message(STATUS "CPU architecture is ARM (AArch64)")
    message(STATUS "Compiler supports NEON.")

    # NEON is automatically available on AArch64
    add_definitions(-DARCH_ARM)
    add_definitions(-DSUPPORTS_NEON)
    set(SOURCE_FILES_AVX2 src/approx_trig_avx2.cpp)

endif()

set(SOURCE_FILES_ALL ${SOURCE_FILES_COMMON} ${SOURCE_FILES_AVX2})

# Define the library
add_library(espressoSq ${SOURCE_FILES_ALL})

# Specify the include directories
target_include_directories(espressoSq PUBLIC ${CMAKE_SOURCE_DIR}/includes)
