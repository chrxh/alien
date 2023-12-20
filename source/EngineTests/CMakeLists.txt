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
    NerveTests.cpp
    NeuronTests.cpp
    SensorTests.cpp
    StatisticsTests.cpp
    Testsuite.cpp
    TransmitterTests.cpp)

target_link_libraries(EngineTests BaseLib)
target_link_libraries(EngineTests EngineGpuKernelsLib)
target_link_libraries(EngineTests EngineImplLib)
target_link_libraries(EngineTests EngineInterfaceLib)

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
