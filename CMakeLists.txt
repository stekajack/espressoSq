cmake_minimum_required(VERSION 3.5.0)
project(espressoSq VERSION 0.1.0 LANGUAGES C CXX)

set(SOURCE_FILES_COMMON src/sq_avx.cpp)
set(SOURCE_FILES_AVX2 "")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")

include(CheckCXXCompilerFlag)

# Check for AVX or AVX2 support first. Rn library only works with x86 intrinsics, but i plan to add arm support with NEON in the future.
check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
if(COMPILER_SUPPORTS_AVX2)
    message(STATUS "Compiler supports AVX2")
    # Check CPU architecture (only if AVX2 is supported)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i[3-6]86")
        message(STATUS "CPU architecture is x86")
        add_definitions(-DSUPPORTS_AVX2)
        add_definitions(-DARCH_X86)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
        set(SOURCE_FILES_AVX2 src/approx_trig_avx2.cpp)

    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64|ARM")
        message(STATUS "CPU architecture is ARM")
        add_definitions(-DARCH_ARM)

        # Check for NEON support (only relevant for ARM CPUs)
        check_cxx_compiler_flag("-mfpu=neon" COMPILER_SUPPORTS_NEON)
        if(COMPILER_SUPPORTS_NEON)
            message(STATUS "Compiler supports NEON. Support for NEON is still not supported.")
            add_definitions(-DSUPPORTS_NEON)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=neon")
        else()
            message(STATUS "Compiler does not support NEON")
        endif()
    else()
        message(STATUS "Unsupported CPU architecture. Compiling generic code.")
    endif()

else()
    message(STATUS "Compiler does not support AVX2. Compiling generic code.")
endif()

set(SOURCE_FILES_ALL ${SOURCE_FILES_COMMON} ${SOURCE_FILES_AVX2})

# Define the library
add_library(espressoSq ${SOURCE_FILES_ALL})

# Specify the include directories
target_include_directories(espressoSq PUBLIC ${CMAKE_SOURCE_DIR}/includes)
