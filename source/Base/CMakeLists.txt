
add_library(Base
    AlienExceptions.cpp
    AlienExceptions.h
    Cache.h
    Definitions.cpp
    Definitions.h
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

target_link_libraries(Base Boost::boost)

if (WIN32)
    target_link_libraries(Base dbghelp)
endif()

target_precompile_headers(Base PRIVATE
    <algorithm>
    <array>
    <chrono>
    <cstdint>
    <filesystem>
    <functional>
    <initializer_list>
    <list>
    <map>
    <memory>
    <mutex>
    <optional>
    <set>
    <sstream>
    <string>
    <unordered_map>
    <unordered_set>
    <variant>
    <vector>
)

if (MSVC)
    target_compile_options(Base PRIVATE "/MP")
endif()
