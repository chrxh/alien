#pragma once

#include <exception>
#include <stdexcept>
#include <string>

// Check for C++23 stacktrace support
// On MSVC, stacktrace is available with C++23
// On GCC/libstdc++, it requires linking with libstdc++_libbacktrace which may not be available
#if defined(_MSC_VER) && _MSC_VER >= 1930
#if __has_include(<stacktrace>)
#include <stacktrace>
#if defined(__cpp_lib_stacktrace) && __cpp_lib_stacktrace >= 202011L
#define ALIEN_HAS_STACKTRACE 1
#endif
#endif
#endif

// Base exception class that captures stack trace at throw site.
// The stack trace is captured in the constructor, preserving the call stack
// at the point where the exception is created/thrown.
// Currently only supported on Windows with MSVC and C++23.
class StackTraceException : public std::runtime_error
{
public:
    StackTraceException(std::string const& what)
        : std::runtime_error(what)
    {
#ifdef ALIEN_HAS_STACKTRACE
        try {
            auto trace = std::stacktrace::current();
            for (auto const& entry : trace) {
                _stackTrace += entry.description() + "\n";
            }
        } catch (...) {
            _stackTrace = "Failed to capture stack trace.";
        }
#endif
    }

    std::string const& getStackTrace() const { return _stackTrace; }

private:
    std::string _stackTrace;
};

class InitialCheckException : public StackTraceException
{
public:
    InitialCheckException(std::string const& what)
        : StackTraceException(what)
    {}
};

class CudaMemoryAllocationException : public StackTraceException
{
public:
    CudaMemoryAllocationException(std::string const& what)
        : StackTraceException(what)
    {}
};
