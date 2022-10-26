target_sources(tests
PUBLIC
    DataTransferTests.cpp
    IntegrationTestFramework.cpp
    IntegrationTestFramework.h
    NerveTests.cpp
    NeuronTests.cpp
    RibosomeTests.cpp
    SensorTests.cpp
    Testsuite.cpp)

target_link_libraries(tests alien_base_lib)
target_link_libraries(tests alien_engine_gpu_kernels_lib)
target_link_libraries(tests alien_engine_impl_lib)
target_link_libraries(tests alien_engine_interface_lib)

target_link_libraries(tests CUDA::cudart_static)
target_link_libraries(tests CUDA::cuda_driver)
target_link_libraries(tests Boost::boost)
target_link_libraries(tests OpenGL::GL OpenGL::GLU)
target_link_libraries(tests GLEW::GLEW)
target_link_libraries(tests glfw)
target_link_libraries(tests glad::glad)
target_link_libraries(tests GTest::GTest GTest::Main)
