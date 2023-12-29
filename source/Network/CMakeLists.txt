
add_library(Network
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
    UserTO.h)

target_link_libraries(Network Base)
target_link_libraries(Network Boost::boost)
    
if (MSVC)
    target_compile_options(Network PRIVATE "/MP")
endif()
