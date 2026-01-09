
add_library(EngineImpl
    DataAccessKernelsService.cu
    DataAccessKernelsService.cuh
    DescriptionConverterService.cpp
    DescriptionConverterService.h
    Definitions.h
    EditKernelsService.cu
    EditKernelsService.cuh
    EngineWorker.cpp
    EngineWorker.h
    GarbageCollectorKernelsService.cu
    GarbageCollectorKernelsService.cuh
    GeometryKernelsService.cu
    GeometryKernelsService.cuh
    SelectionKernelsService.cu
    SelectionKernelsService.cuh
    SimulationCudaFacade.cu
    SimulationCudaFacade.cuh
    SimulationFacadeImpl.cpp
    SimulationFacadeImpl.h
    SimulationKernelsService.cu
    SimulationKernelsService.cuh
    SimulationParametersUpdateService.cu
    SimulationParametersUpdateService.cuh
    StatisticsKernelsService.cu
    StatisticsKernelsService.cuh
    StatisticsService.cu
    StatisticsService.cuh
    TestKernelsService.cu
    TestKernelsService.cuh)

target_link_libraries(EngineImpl Base)
target_link_libraries(EngineImpl EngineGpuKernels)

target_link_libraries(EngineImpl OpenGL::GL)
target_link_libraries(EngineImpl glad::glad)
target_link_libraries(EngineImpl Boost::boost)
target_include_directories(EngineImpl
    PRIVATE
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

target_link_libraries(EngineImpl cudart)

# See https://gitlab.kitware.com/cmake/cmake/-/issues/17520
set_property(TARGET EngineImpl PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

if (MSVC)
    target_compile_options(EngineImpl PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/MP>)
endif()
