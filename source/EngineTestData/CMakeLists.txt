if (MSVC)
    set(LIB_TYPE OBJECT)
else()
    set(LIB_TYPE STATIC)
endif()

add_library(EngineTestData ${LIB_TYPE}
    DescriptionTestDataFactory.cpp
    DescriptionTestDataFactory.h
    TestHelper.cpp
    TestHelper.h)

target_link_libraries(EngineTestData PUBLIC Base)
target_link_libraries(EngineTestData PUBLIC EngineInterface)

target_link_libraries(EngineTestData PUBLIC Boost::boost)

if (MSVC)
    target_compile_options(EngineTestData PRIVATE "/MP")
endif()
