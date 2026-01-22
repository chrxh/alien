target_sources(EngineTests
PUBLIC
    AttackerTests.cpp
    BalanceTests.cpp
    CellStateTransitionTests.cpp
    CommunicatorTests.cpp
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
    MutationTests.cpp
    NeuronTests.cpp
    ObjectConnectionTests.cpp
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

target_include_directories(EngineTests
    PRIVATE
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

target_link_libraries(EngineTests cudart)
target_link_libraries(EngineTests Boost::boost)
target_link_libraries(EngineTests OpenGL::GL OpenGL::GLU)
target_link_libraries(EngineTests GLEW::GLEW)
target_link_libraries(EngineTests glfw)
target_link_libraries(EngineTests glad::glad)
target_link_libraries(EngineTests GTest::GTest GTest::Main)

if (MSVC)
    target_compile_options(EngineTests PRIVATE "/MP")
endif()
