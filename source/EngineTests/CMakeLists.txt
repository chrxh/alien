target_sources(EngineTests
PUBLIC
    CellConnectionTests.cpp
    CellStateTransitionTests.cpp
    ConstructorTests.cpp
    DataTransferTests.cpp
    DefenderTests.cpp
    DescriptionEditTests.cpp
    DetonatorTests.cpp
    EnergyFlowTests.cpp
    GarbageCollectorTests.cpp
    GenomeDescriptionEditServiceTests.cpp
    GenomeDescriptionInfoServiceTests.cpp
    InjectorTests.cpp
    IntegrationTestFramework.cpp
    IntegrationTestFramework.h
    MuscleTests.cpp
    NeuronTests.cpp
    GeneratorTests.cpp
    ParametersEditServiceTests.cpp
    PreviewDescriptionConverterServiceTests.cpp
    ReconnectorTests.cpp
    SensorTests.cpp
    SignalTests.cpp
    StatisticsTests.cpp
    Testsuite.cpp
    DepotTests.cpp)

target_link_libraries(EngineTests Base)
target_link_libraries(EngineTests EngineGpuKernels)
target_link_libraries(EngineTests EngineImpl)
target_link_libraries(EngineTests EngineInterface)
target_link_libraries(EngineTests EngineTestData)

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
