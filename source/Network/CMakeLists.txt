
add_library(Network
    BrowserDataService.cpp
    BrowserDataService.h
    BrowserDataTO.cpp
    BrowserDataTO.h
    Definitions.h
    NetworkService.cpp
    NetworkService.h
    NetworkDataParserService.cpp
    NetworkDataParserService.h
    NetworkDataTO.cpp
    NetworkDataTO.h
    UserTO.h)

target_link_libraries(Network Base)
target_link_libraries(Network Boost::boost)
    
if (MSVC)
    target_compile_options(Network PRIVATE "/MP")
endif()
