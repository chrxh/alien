
add_library(EngineImplLib
    AccessDataTOCache.cpp
    AccessDataTOCache.h
    DescriptionConverter.cpp
    DescriptionConverter.h
    Definitions.h
    EngineWorker.cpp
    EngineWorker.h
    SimulationControllerImpl.cpp
    SimulationControllerImpl.h)

target_link_libraries(EngineImplLib BaseLib)
target_link_libraries(EngineImplLib EngineGpuKernelsLib)

target_link_libraries(EngineImplLib CUDA::cudart_static)
target_link_libraries(EngineImplLib Boost::boost)

if (MSVC)
    target_compile_options(EngineImplLib PRIVATE "/MP")
endif()
