
add_library(EngineTestData
    DescTestDataFactory.cpp
    DescTestDataFactory.h
    TestHelper.cpp
    TestHelper.h)

target_link_libraries(EngineTestData Base)
target_link_libraries(EngineTestData EngineInterface)

target_link_libraries(EngineTestData Boost::boost)

target_precompile_headers(EngineTestData PRIVATE
    <memory>
    <string>
    <vector>
)

if (MSVC)
    target_compile_options(EngineTestData PRIVATE "/MP")
endif()
