cmake_minimum_required(VERSION 3.18)

project(ALiEn LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

set(CMAKE_CUDA_ARCHITECTURES 60)

find_package(Qt6 COMPONENTS Widgets OpenGL OpenGLWidgets Network REQUIRED)
find_package(Boost 1.75 REQUIRED COMPONENTS serialization)

add_subdirectory(external/cuda)
add_subdirectory(external/QJsonModel)
add_subdirectory(source/Base)
add_subdirectory(source/EngineInterface)
add_subdirectory(source/EngineGpu)
add_subdirectory(source/EngineGpuKernels)
add_subdirectory(source/Web)
add_subdirectory(source/Gui)
