#pragma once

#include <string>

#ifdef _WIN32
#include <stacktrace>
#endif

namespace StackTraceHelper
{
#ifdef _WIN32
    // Captures and returns the current stack trace as a formatted string.
    // This function is only available on Windows with C++23 support.
    inline std::string captureStackTrace()
    {
        std::string result;
        try {
            auto trace = std::stacktrace::current();
            for (auto const& entry : trace) {
                result += entry.description() + "\n";
            }
        } catch (...) {
            result = "Failed to capture stack trace.";
        }
        return result;
    }
#else
    // On non-Windows platforms, stack trace capture is not implemented.
    inline std::string captureStackTrace()
    {
        return "Stack trace not available on this platform.";
    }
#endif

    // Logs the exception message along with a stack trace to the logging service.
    void logExceptionWithStackTrace(std::string const& exceptionMessage);
}
