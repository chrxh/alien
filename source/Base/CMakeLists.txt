
add_library(alien_base_lib
    Definitions.cpp
    Definitions.h
    Exceptions.h
    JsonParser.h
    LoggingService.cpp
    LoggingService.h
    Math.cpp
    Math.h
    NumberGenerator.cpp
    NumberGenerator.h
    Physics.cpp
    Physics.h
    Resources.h
    StringHelper.cpp
    StringHelper.h)

target_link_libraries(alien_base_lib Boost::boost)
