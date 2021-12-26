
add_library(alien_base_lib
    BaseServices.cpp
    BaseServices.h
    Definitions.cpp
    Definitions.h
    DllExport.h
    Exceptions.h
    JsonParser.h
    LoggingService.h
    LoggingServiceImpl.cpp
    LoggingServiceImpl.h
    Math.cpp
    Math.h
    NumberGenerator.cpp
    NumberGenerator.h
    Physics.cpp
    Physics.h
    ServiceLocator.cpp
    ServiceLocator.h
    StringFormatter.cpp
    StringFormatter.h)

target_link_libraries(alien_base_lib Boost::boost)
