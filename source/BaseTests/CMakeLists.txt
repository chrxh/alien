target_sources(BaseTests
PUBLIC
    MarkdownParserTests.cpp
    StringHelperTests.cpp
    Testsuite.cpp)

target_link_libraries(BaseTests Base)

target_link_libraries(BaseTests GTest::GTest GTest::Main)

target_precompile_headers(BaseTests PRIVATE
    <string>
    <vector>
    <gtest/gtest.h>
)

if (MSVC)
    target_compile_options(BaseTests PRIVATE "/MP")
endif()
