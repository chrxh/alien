
add_library(Network OBJECT
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

target_include_directories(Network PUBLIC ${CMAKE_SOURCE_DIR}/source ${CMAKE_SOURCE_DIR}/external)
target_link_libraries(Network Boost::boost)
    
if (MSVC)
    target_compile_options(Network PRIVATE "/MP")
endif()
