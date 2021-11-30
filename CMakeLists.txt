
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

add_executable(alien
    source/Gui/FlowGeneratorWindow.cpp
    source/Gui/SimulationScrollbar.cpp
    source/Gui/OpenGLHelper.cpp
    source/Gui/StyleRepository.cpp
    source/Gui/DisplaySettingsDialog.cpp
    source/Gui/NewSimulationDialog.cpp
    source/Gui/FileLogger.cpp
    source/Gui/StartupWindow.cpp
    source/Gui/MainWindow.cpp
    source/Gui/SimpleLogger.cpp
    source/Gui/Viewport.cpp
    source/Gui/SaveSimulationDialog.cpp
    source/Gui/Main.cpp
    source/Gui/AutosaveController.cpp
    source/Gui/Shader.cpp
    source/Gui/GlobalSettings.cpp
    source/Gui/ModeWindow.cpp
    source/Gui/OpenSimulationDialog.cpp
    source/Gui/GettingStartedWindow.cpp
    source/Gui/TemporalControlWindow.cpp
    source/Gui/UiController.cpp
    source/Gui/SpatialControlWindow.cpp
    source/Gui/SimulationView.cpp
    source/Gui/StatisticsWindow.cpp
    source/Gui/AlienImGui.cpp
    source/Gui/LogWindow.cpp
    source/Gui/AboutDialog.cpp
    source/Gui/ColorizeDialog.cpp
    source/Gui/SimulationParametersWindow.cpp
    source/Gui/GpuSettingsDialog.cpp
    source/Gui/EditorController.cpp
    source/Gui/EditorModel.cpp
    source/Gui/SelectionWindow.cpp
    source/Gui/ActionsWindow.cpp
    source/EngineGpuKernels/CudaSimulation.cu
    source/Base/Math.cpp
    source/Base/StringFormatter.cpp
    source/Base/ServiceLocator.cpp
    source/Base/BaseServices.cpp
    source/Base/Physics.cpp
    source/Base/Definitions.cpp
    source/Base/LoggingServiceImpl.cpp
    source/Base/NumberGenerator.cpp
    source/EngineInterface/DescriptionHelper.cpp
    source/EngineInterface/ChangeDescriptions.cpp
    source/EngineInterface/Serializer.cpp
    source/EngineInterface/Descriptions.cpp
    source/EngineInterface/SpaceCalculator.cpp
    source/EngineInterface/Parser.cpp
    #source/EngineInterface/EngineInterfaceSettings.cpp
    source/EngineImpl/SimulationController.cpp
    source/EngineImpl/DataConverter.cpp
    source/EngineImpl/EngineWorker.cpp
    source/EngineImpl/AccessDataTOCache.cpp)

include_directories(
    source
    external)

target_compile_definitions(alien PRIVATE ALIEN_STATIC)

find_package(CUDAToolkit)
target_link_libraries(alien CUDA::cudart_static)
target_link_libraries(alien CUDA::cuda_driver)
    
find_package(Boost REQUIRED COMPONENTS serialization)
target_link_libraries(alien Boost::boost)
target_link_libraries(alien Boost::serialization)

find_package(OpenGL REQUIRED)
target_link_libraries(alien OpenGL::GL OpenGL::GLU)

find_package(GLEW REQUIRED)
target_link_libraries(alien GLEW::GLEW)

find_package(imgui CONFIG REQUIRED)
target_link_libraries(alien imgui::imgui)

find_package(implot CONFIG REQUIRED)
target_link_libraries(alien implot::implot)

find_package(glfw3 CONFIG REQUIRED)
target_link_libraries(alien glfw)

find_package(glad CONFIG REQUIRED)
target_link_libraries(alien glad::glad)

find_path(STB_INCLUDE_DIRS "stb.h")
target_include_directories(alien PRIVATE ${STB_INCLUDE_DIRS})

add_subdirectory(external/ImFileDialog)
target_link_libraries(alien im_file_dialog)

# Copy resources to the build location
add_custom_command(
    TARGET alien POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/msvc/Gui/Resources
        ${CMAKE_CURRENT_BINARY_DIR}/Resources)
