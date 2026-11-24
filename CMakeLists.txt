# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Barın Güzeldemirci

cmake_minimum_required(VERSION 3.16)

file(READ "${CMAKE_CURRENT_SOURCE_DIR}/include/nanovec_version.h" _nanovec_ver_h)
string(REGEX MATCH "#define NANOVEC_VERSION_MAJOR ([0-9]+)" _ "${_nanovec_ver_h}")
set(NANOVEC_VERSION_MAJOR "${CMAKE_MATCH_1}")
string(REGEX MATCH "#define NANOVEC_VERSION_MINOR ([0-9]+)" _ "${_nanovec_ver_h}")
set(NANOVEC_VERSION_MINOR "${CMAKE_MATCH_1}")
string(REGEX MATCH "#define NANOVEC_VERSION_PATCH ([0-9]+)" _ "${_nanovec_ver_h}")
set(NANOVEC_VERSION_PATCH "${CMAKE_MATCH_1}")

project(nanovec
    VERSION ${NANOVEC_VERSION_MAJOR}.${NANOVEC_VERSION_MINOR}.${NANOVEC_VERSION_PATCH}
    LANGUAGES C
)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(NANOVEC_BUILD_TESTS "Build C test targets" ON)
option(NANOVEC_BUILD_TOOLS "Build CLI and benchmark tools" ON)
option(NANOVEC_ENABLE_AVX2 "Compile AVX2 SIMD kernels on x86_64" ON)
option(NANOVEC_ENABLE_AVX512 "Compile AVX-512 SIMD kernels on x86_64" ON)

find_package(Threads REQUIRED)
include(GNUInstallDirs)

set(NANOVEC_COMMON_FLAGS
    -Wall
    -Wextra
    -Wpedantic
    -fPIC
    -fvisibility=hidden
    -D_GNU_SOURCE
)

set(NANOVEC_OPT_FLAGS
    $<$<CONFIG:Release>:-O3>
    $<$<CONFIG:RelWithDebInfo>:-O3 -g>
    $<$<CONFIG:Debug>:-O0 -g3 -fno-omit-frame-pointer>
)

file(GLOB NANOVEC_ALL_SRCS CONFIGURE_DEPENDS
    src/core/*.c
    src/math/*.c
    src/quant/*.c
    src/storage/*.c
    src/index/*.c
    src/inference/*.c
    src/platform/*.c
)

set(NANOVEC_LIB_SRCS ${NANOVEC_ALL_SRCS})
list(FILTER NANOVEC_LIB_SRCS EXCLUDE REGEX "src/math/simd_(avx2|avx512|neon)\\.c$")

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
    if(NANOVEC_ENABLE_AVX2 AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/math/simd_avx2.c")
        list(APPEND NANOVEC_LIB_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/src/math/simd_avx2.c")
        set_source_files_properties(
            "${CMAKE_CURRENT_SOURCE_DIR}/src/math/simd_avx2.c"
            PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma"
        )
    endif()
    if(NANOVEC_ENABLE_AVX512 AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/math/simd_avx512.c")
        list(APPEND NANOVEC_LIB_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/src/math/simd_avx512.c")
        set_source_files_properties(
            "${CMAKE_CURRENT_SOURCE_DIR}/src/math/simd_avx512.c"
            PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512dq;-mavx512bw"
        )
    endif()
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|arm)")
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/math/simd_neon.c")
        list(APPEND NANOVEC_LIB_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/src/math/simd_neon.c")
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "^arm" AND NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^arm64")
            set_source_files_properties(
                "${CMAKE_CURRENT_SOURCE_DIR}/src/math/simd_neon.c"
                PROPERTIES COMPILE_OPTIONS "-mfpu=neon"
            )
        endif()
    endif()
endif()

if(NANOVEC_LIB_SRCS)
    add_library(nanovec_objs OBJECT ${NANOVEC_LIB_SRCS})
    target_include_directories(nanovec_objs PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
    )
    target_compile_options(nanovec_objs PRIVATE ${NANOVEC_COMMON_FLAGS} ${NANOVEC_OPT_FLAGS})

    add_library(nanovec_static STATIC $<TARGET_OBJECTS:nanovec_objs>)
    set_target_properties(nanovec_static PROPERTIES OUTPUT_NAME nanovec)
    target_include_directories(nanovec_static PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )
    target_link_libraries(nanovec_static PUBLIC Threads::Threads m)

    add_library(nanovec_shared SHARED $<TARGET_OBJECTS:nanovec_objs>)
    set_target_properties(nanovec_shared PROPERTIES
        OUTPUT_NAME nanovec
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
    )
    target_include_directories(nanovec_shared PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )
    target_link_libraries(nanovec_shared PUBLIC Threads::Threads m)

    if(NANOVEC_BUILD_TOOLS)
        file(GLOB NANOVEC_TOOL_SRCS CONFIGURE_DEPENDS tools/*.c)
        foreach(tool_src ${NANOVEC_TOOL_SRCS})
            get_filename_component(tool_name "${tool_src}" NAME_WE)
            add_executable(${tool_name} "${tool_src}")
            target_include_directories(${tool_name} PRIVATE
                "${CMAKE_CURRENT_SOURCE_DIR}/include"
                "${CMAKE_CURRENT_SOURCE_DIR}/src"
            )
            target_compile_options(${tool_name} PRIVATE ${NANOVEC_COMMON_FLAGS} ${NANOVEC_OPT_FLAGS})
            target_link_libraries(${tool_name} PRIVATE nanovec_static)
        endforeach()
    endif()

    if(NANOVEC_BUILD_TESTS)
        include(CTest)
        file(GLOB NANOVEC_TEST_SRCS CONFIGURE_DEPENDS tests/test_*.c)
        foreach(test_src ${NANOVEC_TEST_SRCS})
            get_filename_component(test_name "${test_src}" NAME_WE)
            add_executable(${test_name} "${test_src}")
            target_include_directories(${test_name} PRIVATE
                "${CMAKE_CURRENT_SOURCE_DIR}/include"
                "${CMAKE_CURRENT_SOURCE_DIR}/src"
            )
            target_compile_options(${test_name} PRIVATE ${NANOVEC_COMMON_FLAGS} ${NANOVEC_OPT_FLAGS})
            target_link_libraries(${test_name} PRIVATE nanovec_static)
            add_test(NAME ${test_name} COMMAND ${test_name})
        endforeach()
    endif()

    install(TARGETS nanovec_static nanovec_shared
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
endif()

file(GLOB NANOVEC_PUBLIC_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h")
install(FILES ${NANOVEC_PUBLIC_HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
