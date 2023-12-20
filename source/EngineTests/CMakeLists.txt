target_sources(engine_tests
PUBLIC
    AttackerTests.cpp
    CellConnectionTests.cpp
    ConstructorTests.cpp
    DataTransferTests.cpp
    DefenderTests.cpp
    DescriptionHelperTests.cpp
    DetonatorTests.cpp
    InjectorTests.cpp
    IntegrationTestFramework.cpp
    IntegrationTestFramework.h
    LivingStateTransitionTests.cpp
    MuscleTests.cpp
    MutationTests.cpp
    NerveTests.cpp
    NeuronTests.cpp
    SensorTests.cpp
    StatisticsTests.cpp
    Testsuite.cpp
    TransmitterTests.cpp)

target_link_libraries(engine_tests alien_base_lib)
target_link_libraries(engine_tests alien_engine_gpu_kernels_lib)
target_link_libraries(engine_tests alien_engine_impl_lib)
target_link_libraries(engine_tests alien_engine_interface_lib)

target_link_libraries(engine_tests CUDA::cudart_static)
target_link_libraries(engine_tests CUDA::cuda_driver)
target_link_libraries(engine_tests Boost::boost)
target_link_libraries(engine_tests OpenGL::GL OpenGL::GLU)
target_link_libraries(engine_tests GLEW::GLEW)
target_link_libraries(engine_tests glfw)
target_link_libraries(engine_tests glad::glad)
target_link_libraries(engine_tests GTest::GTest GTest::Main)

if (MSVC)
    target_compile_options(engine_tests PRIVATE "/MP")
endif()
