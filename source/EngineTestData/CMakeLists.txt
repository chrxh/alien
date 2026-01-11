
add_library(EngineTestData OBJECT
    DescriptionTestDataFactory.cpp
    DescriptionTestDataFactory.h
    TestHelper.cpp
    TestHelper.h)

target_include_directories(EngineTestData PUBLIC ${CMAKE_SOURCE_DIR}/source ${CMAKE_SOURCE_DIR}/external)
target_link_libraries(EngineTestData Boost::boost)

if (MSVC)
    target_compile_options(EngineTestData PRIVATE "/MP")
endif()
