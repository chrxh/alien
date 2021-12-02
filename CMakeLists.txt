
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

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -lineinfo --use-local-env")

project(alien-project LANGUAGES C CXX CUDA)

include_directories(
    source
    external)

add_compile_definitions(ALIEN_STATIC)

add_executable(alien)

find_package(CUDAToolkit)
find_package(Boost REQUIRED COMPONENTS serialization)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(implot CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(glad CONFIG REQUIRED)

add_subdirectory(external/ImFileDialog)
add_subdirectory(source/Base)
add_subdirectory(source/EngineGpuKernels)
add_subdirectory(source/EngineImpl)
add_subdirectory(source/EngineInterface)
add_subdirectory(source/Gui)

# Copy resources to the build location
add_custom_command(
    TARGET alien POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/Resources
        ${CMAKE_CURRENT_BINARY_DIR}/Resources)

# Copy imgui.ini
add_custom_command(
    TARGET alien POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_SOURCE_DIR}/imgui.ini
        ${CMAKE_CURRENT_BINARY_DIR})
