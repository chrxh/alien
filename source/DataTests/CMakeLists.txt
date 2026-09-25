target_sources(DataTests
PUBLIC
    DescEditServiceTests.cpp
    DescValidationServiceTests.cpp
    GenomeDescAccessServiceTests.cpp
    GenomeDescEditServiceTests.cpp
    LocationEditServiceTests.cpp
    ParametersAccessServiceTests.cpp
    PreviewDescConverterServiceTests.cpp
    SpecificationFilterServiceTests.cpp
    Testsuite.cpp)

target_link_libraries(DataTests Base)
target_link_libraries(DataTests Data)
target_link_libraries(DataTests EngineTestData)

target_link_libraries(DataTests Boost::boost)
target_link_libraries(DataTests GTest::GTest GTest::Main)

target_precompile_headers(DataTests PRIVATE
    <algorithm>
    <memory>
    <string>
    <vector>
    <gtest/gtest.h>
)

if (MSVC)
    target_compile_options(DataTests PRIVATE "/MP")
endif()
