target_sources(NetworkTests
PUBLIC
    McpArgumentsTests.cpp
    McpJsonTests.cpp
    McpServerTests.cpp
    NetworkResourceServiceTests.cpp
    Testsuite.cpp)

target_link_libraries(NetworkTests Base)
target_link_libraries(NetworkTests Data)
target_link_libraries(NetworkTests EngineInterface)
target_link_libraries(NetworkTests Network)

target_link_libraries(NetworkTests Boost::boost)
target_link_libraries(NetworkTests GTest::GTest GTest::Main)

target_precompile_headers(NetworkTests PRIVATE
    <memory>
    <string>
    <vector>
    <gtest/gtest.h>
)

if (MSVC)
    target_compile_options(NetworkTests PRIVATE "/MP")
endif()
