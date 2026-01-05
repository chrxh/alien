
add_library(Base
    Cache.h
    Definitions.cpp
    Definitions.h
    Exceptions.h
    ExitScopeGuard.h
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
    MathTypes.cpp
    MathTypes.h
    Physics.cpp
    Physics.h
    Resources.h
    Singleton.h
    StringHelper.cpp
    StringHelper.h
    UnlockGuard.h
    VersionParserService.cpp
    VersionParserService.h)

if (MSVC)
    target_compile_options(Base PRIVATE "/MP")
endif()
