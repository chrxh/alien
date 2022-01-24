
add_library(alien_engine_impl_lib
    AccessDataTOCache.cpp
    AccessDataTOCache.h
    DataConverter.cpp
    DataConverter.h
    Definitions.h
    EngineWorker.cpp
    EngineWorker.h
    SimulationControllerImpl.cpp
    SimulationControllerImpl.h)

target_link_libraries(alien_engine_impl_lib alien_base_lib)
target_link_libraries(alien_engine_impl_lib alien_engine_gpu_kernels_lib)

target_link_libraries(alien_engine_impl_lib CUDA::cudart_static)
target_link_libraries(alien_engine_impl_lib Boost::boost)

