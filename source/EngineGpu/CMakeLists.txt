project(EngineGpu)

set(EngineGpu_SOURCES
    CudaController.cpp
    CudaController.h
    CudaJobs.h
    CudaWorker.cpp
    CudaWorker.h
    DataConverter.cpp
    DataConverter.h
    Definitions.h
    DefinitionsImpl.h
    DllExport.h
    EngineGpuBuilderFacade.h
    EngineGpuBuilderFacadeImpl.cpp
    EngineGpuBuilderFacadeImpl.h
    EngineGpuData.cpp
    EngineGpuData.h
    EngineGpuServices.cpp
    EngineGpuServices.h
    EngineGpuSettings.cpp
    EngineGpuSettings.h
    SimulationAccessGpu.h
    SimulationAccessGpuImpl.cpp
    SimulationAccessGpuImpl.h
    SimulationContextGpuImpl.cpp
    SimulationContextGpuImpl.h
    SimulationControllerGpu.h
    SimulationControllerGpuImpl.cpp
    SimulationControllerGpuImpl.h
    SimulationMonitorGpu.h
    SimulationMonitorGpuImpl.cpp
    SimulationMonitorGpuImpl.h
)

add_library(EngineGpu SHARED ${EngineGpu_SOURCES})
add_library(ALiEn::EngineGpu ALIAS EngineGpu)

target_compile_definitions(EngineGpu PRIVATE ENGINEGPU_LIB)

target_link_libraries(EngineGpu PUBLIC Qt6::Widgets Qt6::OpenGL)

target_include_directories(EngineGpu
PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/.."
PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
