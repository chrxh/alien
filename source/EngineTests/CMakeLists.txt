target_sources(EngineTests
PUBLIC
    AttackerTests.cpp
    BalanceTests.cpp
    CellConnectionTests.cpp
    CellStateTransitionTests.cpp
    ConstructorTests.cpp
    CreatureTests.cpp
    DataTransferTests.cpp
    DefenderTests.cpp
    DepotTests.cpp
    DescriptionEditTests.cpp
    DetonatorTests.cpp
    EditTests.cpp
    DigestorTests.cpp
    EnergyFlowTests.cpp
    EnergyParticleTests.cpp
    FrontAngleUpdateTests.cpp
    GarbageCollectorTests.cpp
    GeneratorTests.cpp
    GeometryTests.cpp
    InjectorTests.cpp
    IntegrationTestFramework.cpp
    IntegrationTestFramework.h
    LayerParameterTests.cpp
    MemoryTests.cpp
    MuscleTests.cpp
    NeuronTests.cpp
    PhysicsTests.cpp
    RadiationTests.cpp
    ReconnectorTests.cpp
    SensorTests.cpp
    SignalTests.cpp
    StatisticsTests.cpp
    Testsuite.cpp)

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
