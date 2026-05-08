cmake_minimum_required(VERSION 3.16)
project(transformers-lite
    VERSION 0.1.0
    LANGUAGES CXX
    DESCRIPTION "Transformer inference in C++"
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "")

option(BUILD_TESTS "Build unit tests" OFF)
option(ENABLE_ASAN "Enable AddressSanitizer" ON)
option(ENABLE_CUDA "Enable CUDA compute backend" OFF)
option(ENABLE_AVX512 "Enable AVX-512 compute backend" OFF)

find_package(OpenMP)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

file(GLOB_RECURSE HDR_FILES ${PROJECT_SOURCE_DIR}/include/transformers-lite/*.hpp)

add_library(transformers_lite INTERFACE)
target_sources(transformers_lite INTERFACE ${HDR_FILES})
target_include_directories(transformers_lite
    INTERFACE
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

if(ENABLE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_compile_definitions(transformers_lite INTERFACE TRANSFORMERS_CUDA_ENABLED)
    target_link_libraries(transformers_lite INTERFACE CUDA::cudart)
endif()

if(ENABLE_AVX512)
    target_compile_definitions(transformers_lite INTERFACE TRANSFORMERS_AVX512_ENABLED)
    target_compile_options(transformers_lite INTERFACE -mavx512f)
endif()

if(ENABLE_ASAN)
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address)
endif()

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
