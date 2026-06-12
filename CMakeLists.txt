
cmake_minimum_required(VERSION 3.31 FATAL_ERROR)

# Using the vcpkg submodule
set(CMAKE_TOOLCHAIN_FILE external/vcpkg/scripts/buildsystems/vcpkg.cmake CACHE STRING "")

# Default to Debug for single-config generators. Multi-config generators (Visual
# Studio, Ninja Multi-Config, Xcode) select the configuration at build time, so
# CMAKE_BUILD_TYPE must stay empty for them.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_GENERATOR MATCHES "Multi-Config|Visual Studio|Xcode")
    message("Build type not specified, using Debug")
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type not specified, using Debug" FORCE)
endif()

# Default CUDA target architectures
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo --use-local-env -use_fast_math")

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --threads 0")
endif()

project(alien-project LANGUAGES C CXX CUDA)

include_directories(
    source
    external)

if (MSVC)
    # Suppress `warning C4005: 'FOO': macro redefinition`
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-wd4005>)
    
    # Treat warnings as errors
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-WX>)

    # Use standard conforming preprocessor (required by CUDA CCCL headers since CUDA 13.2)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>)
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>)
else()
    # Treat warnings as errors
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Werror>)
endif()
add_compile_definitions($<$<COMPILE_LANGUAGE:CXX>:NOMINMAX>)

# Treat all NVCC (CUDA) warnings as errors
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--Werror=all-warnings>)

# Emit host debug info (nvcc -g) only for debug-capable configurations. On MSVC,
# -g forces the host compiler to generate debug info that is serialized through
# mspdbsrv, which noticeably slows down Release compilation for no runtime benefit.
add_compile_options($<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug,RelWithDebInfo>>:-g>)

add_executable(alien)
add_executable(cli)
add_executable(EngineTests)
add_executable(EngineInterfaceTests)
add_executable(NetworkTests)
add_executable(PersisterTests)

# find_package(CUDAToolkit REQUIRED) is too slow
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
add_subdirectory(source/EngineKernels)
add_subdirectory(source/EngineImpl)
add_subdirectory(source/EngineInterface)
add_subdirectory(source/EngineInterfaceTests)
add_subdirectory(source/EngineTestData)
add_subdirectory(source/EngineTests)
add_subdirectory(source/Gui)
add_subdirectory(source/Network)
add_subdirectory(source/NetworkTests)
add_subdirectory(source/PersisterImpl)
add_subdirectory(source/PersisterInterface)
add_subdirectory(source/PersisterTests)
add_subdirectory(source/Shaders)

# Copy resources and imgui.ini next to the build output so that ALIEN finds them
add_custom_command(
    TARGET alien POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/resources
        ${CMAKE_CURRENT_BINARY_DIR}/resources
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/resources
        $<TARGET_FILE_DIR:alien>/resources)

add_custom_command(
    TARGET alien POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_SOURCE_DIR}/imgui.ini
        ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_SOURCE_DIR}/imgui.ini
        $<TARGET_FILE_DIR:alien>)
