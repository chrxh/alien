target_sources(EngineInterfaceTests
PUBLIC
    ExtinctLineageAccumulatorTests.cpp
    Testsuite.cpp)

target_link_libraries(EngineInterfaceTests Base)
target_link_libraries(EngineInterfaceTests Data)
target_link_libraries(EngineInterfaceTests EngineInterface)

target_link_libraries(EngineInterfaceTests Boost::boost)
target_link_libraries(EngineInterfaceTests GTest::GTest GTest::Main)

target_precompile_headers(EngineInterfaceTests PRIVATE
    <algorithm>
    <memory>
    <string>
    <vector>
    <gtest/gtest.h>
)

if (MSVC)
    target_compile_options(EngineInterfaceTests PRIVATE "/MP")
endif()
