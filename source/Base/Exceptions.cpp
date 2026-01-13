#include "Exceptions.h"

// Check for C++23 stacktrace support
// This file is compiled with C++23 (not CUDA), so stacktrace should be available
#if defined(_MSC_VER) && _MSC_VER >= 1930
#if __has_include(<stacktrace>)
#include <stacktrace>
#if defined(__cpp_lib_stacktrace) && __cpp_lib_stacktrace >= 202011L
#define ALIEN_HAS_STACKTRACE 1
#endif
#endif
#endif

std::string captureStackTrace()
{
    std::string result;
#ifdef ALIEN_HAS_STACKTRACE
    try {
        auto trace = std::stacktrace::current();
        for (auto const& entry : trace) {
            result += entry.description() + "\n";
        }
    } catch (...) {
        result = "Failed to capture stack trace.";
    }
#endif
    return result;
}
