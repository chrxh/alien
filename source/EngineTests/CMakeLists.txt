target_sources(EngineTests
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
    NeuronTests.cpp
    NerveTests_New.cpp
    ReconnectorTests.cpp
    SensorTests.cpp
    SignalTests_New.cpp
    StatisticsTests.cpp
    Testsuite.cpp
    TransmitterTests.cpp)

target_link_libraries(EngineTests Base)
target_link_libraries(EngineTests EngineGpuKernels)
target_link_libraries(EngineTests EngineImpl)
target_link_libraries(EngineTests EngineInterface)

target_link_libraries(EngineTests CUDA::cudart_static)
target_link_libraries(EngineTests CUDA::cuda_driver)
target_link_libraries(EngineTests Boost::boost)
target_link_libraries(EngineTests OpenGL::GL OpenGL::GLU)
target_link_libraries(EngineTests GLEW::GLEW)
target_link_libraries(EngineTests glfw)
target_link_libraries(EngineTests glad::glad)
target_link_libraries(EngineTests GTest::GTest GTest::Main)

if (MSVC)
    target_compile_options(EngineTests PRIVATE "/MP")
endif()
