#include "Exceptions.h"

// Check for C++23 stacktrace support on Windows (MSVC)
#if defined(_MSC_VER) && _MSC_VER >= 1930
#if __has_include(<stacktrace>)
#include <stacktrace>
#if defined(__cpp_lib_stacktrace) && __cpp_lib_stacktrace >= 202011L
#define ALIEN_HAS_STACKTRACE 1
#endif
#endif
#endif

// Linux stack trace support using execinfo.h
#if defined(__linux__) && !defined(ALIEN_HAS_STACKTRACE)
#include <cstdlib>
#include <memory>
#include <sstream>

#include <cxxabi.h>
#include <execinfo.h>

#define ALIEN_HAS_LINUX_BACKTRACE 1
#endif

std::string captureStackTrace()
{
    std::string result;

#ifdef ALIEN_HAS_STACKTRACE
    // Windows: Use C++23 std::stacktrace
    try {
        auto trace = std::stacktrace::current();
        for (auto const& entry : trace) {
            result += entry.description() + "\n";
        }
    } catch (...) {
        result = "Failed to capture stack trace.";
    }

#elif defined(ALIEN_HAS_LINUX_BACKTRACE)
    // Linux: Use backtrace() and backtrace_symbols()
    constexpr int maxFrames = 64;
    void* frames[maxFrames];

    int numFrames = backtrace(frames, maxFrames);
    char** symbols = backtrace_symbols(frames, numFrames);

    if (symbols) {
        std::ostringstream oss;
        for (int i = 0; i < numFrames; ++i) {
            // Try to demangle C++ symbol names
            std::string symbol(symbols[i]);

            // Parse the symbol to extract the mangled name
            // Format: ./program(mangledName+0x123) [0x456789]
            size_t beginName = symbol.find('(');
            size_t endName = symbol.find('+', beginName);

            if (beginName != std::string::npos && endName != std::string::npos) {
                std::string mangledName = symbol.substr(beginName + 1, endName - beginName - 1);

                int status = 0;
                char* demangled = abi::__cxa_demangle(mangledName.c_str(), nullptr, nullptr, &status);

                if (status == 0 && demangled) {
                    oss << "[" << i << "] " << demangled << "\n";
                    free(demangled);
                } else {
                    oss << "[" << i << "] " << symbol << "\n";
                }
            } else {
                oss << "[" << i << "] " << symbol << "\n";
            }
        }
        free(symbols);
        result = oss.str();
    } else {
        result = "Failed to capture stack trace.";
    }
#endif

    return result;
}
