target_sources(McpToolsTests
PUBLIC
    GenomeMassEditServiceTests.cpp
    JsonPathTests.cpp
    McpInspectionToolsTests.cpp
    Testsuite.cpp)

target_link_libraries(McpToolsTests Base)
target_link_libraries(McpToolsTests Data)
target_link_libraries(McpToolsTests McpToolsImpl)
target_link_libraries(McpToolsTests PersisterInterface)

target_link_libraries(McpToolsTests Boost::boost)
target_link_libraries(McpToolsTests Boost::json)
target_link_libraries(McpToolsTests GTest::GTest GTest::Main)

target_precompile_headers(McpToolsTests PRIVATE
    <memory>
    <string>
    <vector>
    <gtest/gtest.h>
)

if (MSVC)
    target_compile_options(McpToolsTests PRIVATE "/MP")
endif()
