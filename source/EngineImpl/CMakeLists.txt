
add_library(EngineImpl
    DataAccessKernelsService.cu
    DataAccessKernelsService.h
    DescriptionConverterService.cpp
    DescriptionConverterService.h
    Definitions.h
    EditKernelsService.cu
    EditKernelsService.h
    EngineWorker.cpp
    EngineWorker.h
    GarbageCollectorKernelsService.cu
    GarbageCollectorKernelsService.h
    GeometryKernelsService.cu
    GeometryKernelsService.h
    SelectionKernelsService.cu
    SelectionKernelsService.h
    SimulationCudaFacade.cu
    SimulationCudaFacade.h
    SimulationFacadeImpl.cpp
    SimulationFacadeImpl.h
    SimulationKernelsService.cu
    SimulationKernelsService.h
    SimulationParametersUpdateService.cu
    SimulationParametersUpdateService.h
    StatisticsKernelsService.cu
    StatisticsKernelsService.h
    StatisticsService.cu
    StatisticsService.h
    TestKernelsService.cu
    TestKernelsService.h)

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
    target_compile_options(EngineImpl PRIVATE "/MP")
endif()
