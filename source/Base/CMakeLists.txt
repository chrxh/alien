
add_library(Base
    Definitions.cpp
    Definitions.h
    Exceptions.h
    FileLogger.cpp
    FileLogger.h
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
    Vector2D.h
    VersionChecker.cpp
    VersionChecker.h)

target_link_libraries(Base Boost::boost)

if (MSVC)
    target_compile_options(Base PRIVATE "/MP")
endif()
