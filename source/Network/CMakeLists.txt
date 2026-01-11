if (MSVC)
    set(LIB_TYPE OBJECT)
else()
    set(LIB_TYPE STATIC)
endif()

add_library(Network ${LIB_TYPE}
    Definitions.h
    NetworkService.cpp
    NetworkService.h
    NetworkResourceParserService.cpp
    NetworkResourceParserService.h
    NetworkResourceRawTO.cpp
    NetworkResourceRawTO.h
    NetworkResourceService.cpp
    NetworkResourceService.h
    NetworkResourceTreeTO.cpp
    NetworkResourceTreeTO.h
    NetworkValidationService.cpp
    NetworkValidationService.h
    UserTO.h)

target_link_libraries(Network PUBLIC Base)
target_link_libraries(Network PUBLIC Boost::boost)
    
if (MSVC)
    target_compile_options(Network PRIVATE "/MP")
endif()
