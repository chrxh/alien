project(EngineGpuKernels)

set(EngineGpuKernels_SOURCES
    CudaSimulation.cu
)

add_library(EngineGpuKernels SHARED ${EngineGpuKernels_SOURCES})
add_library(ALiEn::EngineGpuKernels ALIAS EngineGpuKernels)

target_compile_definitions(EngineGpuKernels PRIVATE ENGINEGPUKERNELS_LIB)

target_link_libraries(EngineGpuKernels PRIVATE CudaHelper)

target_include_directories(EngineGpuKernels
PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/.."
PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
