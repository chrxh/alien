
add_library(NetworkLib
    BrowserDataService.cpp
    BrowserDataService.h
    BrowserDataTO.cpp
    BrowserDataTO.h
    Definitions.h
    NetworkController.cpp
    NetworkController.h
    NetworkDataParserService.cpp
    NetworkDataParserService.h
    NetworkDataTO.cpp
    NetworkDataTO.h
    UserTO.h)

target_link_libraries(NetworkLib BaseLib)
target_link_libraries(NetworkLib Boost::boost)
    
if (MSVC)
    target_compile_options(NetworkLib PRIVATE "/MP")
endif()
