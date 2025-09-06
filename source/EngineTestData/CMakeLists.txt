
add_library(EngineTestData
    DescriptionTestDataFactory.cpp
    DescriptionTestDataFactory.h)

target_link_libraries(EngineTestData Base)
target_link_libraries(EngineTestData EngineInterface)

target_link_libraries(EngineTestData Boost::boost)

if (MSVC)
    target_compile_options(EngineTestData PRIVATE "/MP")
endif()
