
add_library(EngineImpl
    DescriptionConverterService.cpp
    DescriptionConverterService.h
    Definitions.h
    EngineWorker.cpp
    EngineWorker.h
    SimulationFacadeImpl.cpp
    SimulationFacadeImpl.h)

target_link_libraries(EngineImpl Base)
target_link_libraries(EngineImpl EngineGpuKernels)

target_link_libraries(EngineImpl OpenGL::GL)
target_link_libraries(EngineImpl CUDA::cudart_static)

if (MSVC)
    target_compile_options(EngineImpl PRIVATE "/MP")
endif()
