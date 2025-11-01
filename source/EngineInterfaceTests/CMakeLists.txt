target_sources(EngineInterfaceTests
PUBLIC
    GenomeDescriptionEditServiceTests.cpp
    GenomeDescriptionInfoServiceTests.cpp
    ParametersEditServiceTests.cpp
    PreviewDescriptionConverterServiceTests.cpp
    TestHelper.h
    Testsuite.cpp)

target_link_libraries(EngineInterfaceTests Base)
target_link_libraries(EngineInterfaceTests EngineInterface)

target_link_libraries(EngineInterfaceTests Boost::boost)
target_link_libraries(EngineInterfaceTests GTest::GTest GTest::Main)

if (MSVC)
    target_compile_options(EngineInterfaceTests PRIVATE "/MP")
endif()
