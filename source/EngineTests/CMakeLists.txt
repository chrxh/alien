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
    HeadUpdateTests.cpp
    GarbageCollectorTests.cpp
    GeneratorTests.cpp
    GeometryTests.cpp
    InjectorTests.cpp
    IntegrationTestFramework.cpp
    IntegrationTestFramework.h
    LayerParameterTests.cpp
    MemoryTests.cpp
    MetaMutationTests.cpp
    MuscleTests.cpp
    MutationTests.cpp
    MutationTestsBase.h
    NeuronTests.cpp
    ObjectConnectionTests.cpp
    PhysicsTests.cpp
    RadiationTests.cpp
    ReconnectorTests.cpp
    SensorTests.cpp
    SimulationControlTests.cpp
    StatisticsTests.cpp
    Testsuite.cpp
    VoidTests.cpp)

target_link_libraries(EngineTests Base)
target_link_libraries(EngineTests EngineKernels)
target_link_libraries(EngineTests EngineImpl)
target_link_libraries(EngineTests EngineInterface)
target_link_libraries(EngineTests PersisterInterface)
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

target_precompile_headers(EngineTests PRIVATE
    <algorithm>
    <cmath>
    <memory>
    <ranges>
    <string>
    <vector>
    <gtest/gtest.h>
)

if (MSVC)
    target_compile_options(EngineTests PRIVATE "/MP")
endif()
