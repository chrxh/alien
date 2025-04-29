target_sources(EngineTests
PUBLIC
    AttackerTests.cpp
    CellConnectionTests_New.cpp
    ConstructorTests.cpp
    ConstructorTests_New.cpp
    DataTransferTests.cpp
    DefenderTests.cpp
    DescriptionEditTests.cpp
    DetonatorTests.cpp
    EnergyFlowTests_New.cpp
    InjectorTests.cpp
    IntegrationTestFramework.cpp
    IntegrationTestFramework.h
    LivingStateTransitionTests.cpp
    MuscleTests.cpp
    MuscleTests_New.cpp
    MutationTests.cpp
    NeuronTests.cpp
    OscillatorTests_New.cpp
    ParametersEditServiceTests_New.cpp
    ReconnectorTests.cpp
    SensorTests.cpp
    SensorTests_New.cpp
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
