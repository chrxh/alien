
cmake_minimum_required(VERSION 3.17 FATAL_ERROR)

# Using the vcpkg submodule
set(CMAKE_TOOLCHAIN_FILE external/vcpkg/scripts/buildsystems/vcpkg.cmake CACHE STRING "")

# Default to Debug
if(NOT CMAKE_BUILD_TYPE)
    message("Build type not specified, using Debug")
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type not specified, using Debug" FORCE)
endif(NOT CMAKE_BUILD_TYPE)

# Default CUDA target architectures
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 60)
endif()

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -lineinfo --use-local-env -use_fast_math")

project(alien-project LANGUAGES C CXX CUDA)

include_directories(
    source
    external)

if (MSVC)
    # Suppress `warning C4005: 'FOO': macro redefinition`
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-wd4005>)
    
    # Treat warnings as errors
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-WX>)
else()
    # Treat warnings as errors
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Werror>)
endif()
add_compile_definitions($<$<COMPILE_LANGUAGE:CXX>:NOMINMAX>)

# Treat all NVCC (CUDA) warnings as errors
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--Werror=all-warnings>)

add_executable(alien)
add_executable(cli)
add_executable(EngineTests)
add_executable(NetworkTests)

find_package(CUDAToolkit)
find_package(Boost REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(cereal CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(implot CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(glad CONFIG REQUIRED)
find_package(GTest REQUIRED)
find_package(ZLIB REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(CLI11 CONFIG REQUIRED)

add_subdirectory(external/ImFileDialog)
add_subdirectory(source/Base)
add_subdirectory(source/Cli)
add_subdirectory(source/EngineGpuKernels)
add_subdirectory(source/EngineImpl)
add_subdirectory(source/EngineInterface)
add_subdirectory(source/EngineTests)
add_subdirectory(source/Gui)
add_subdirectory(source/Network)
add_subdirectory(source/NetworkTests)
add_subdirectory(source/PersisterImpl)
add_subdirectory(source/PersisterInterface)

# Copy resources to the build location
add_custom_command(
    TARGET alien POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/resources
        ${CMAKE_CURRENT_BINARY_DIR}/resources)

# Copy imgui.ini
add_custom_command(
    TARGET alien POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_SOURCE_DIR}/imgui.ini
        ${CMAKE_CURRENT_BINARY_DIR})
