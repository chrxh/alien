
add_library(alien_base_lib
    Definitions.cpp
    Definitions.h
    Exceptions.h
    GlobalSettings.cpp
    GlobalSettings.h
    Hashes.h
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
    StringHelper.h
    Vector2D.cpp
    Vector2D.h)

target_link_libraries(alien_base_lib Boost::boost)

if (MSVC)
    target_compile_options(alien_base_lib PRIVATE "/MP")
endif()
