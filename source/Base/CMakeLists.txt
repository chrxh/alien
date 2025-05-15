
add_library(Base
    Cache.h
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
    Macros.h
    Math.cpp
    Math.h
    NumberGenerator.cpp
    NumberGenerator.h
    OutOfScopeGuard.h
    Physics.cpp
    Physics.h
    Resources.h
    Singleton.h
    StringHelper.cpp
    StringHelper.h
    UnlockGuard.h
    Vector2D.cpp
    Vector2D.h
    VersionParserService.cpp
    VersionParserService.h)

target_link_libraries(Base Boost::boost)

if (MSVC)
    target_compile_options(Base PRIVATE "/MP")
endif()
