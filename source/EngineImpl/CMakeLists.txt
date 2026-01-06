
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
target_link_libraries(EngineImpl Boost::boost)
target_include_directories(EngineImpl
    PRIVATE
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

target_link_libraries(EngineImpl cudart)


if (MSVC)
    target_compile_options(EngineImpl PRIVATE "/MP")
endif()
