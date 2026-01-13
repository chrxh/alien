#pragma once

#include <exception>
#include <stdexcept>
#include <string>

#ifdef _WIN32
#include <stacktrace>
#endif

// Base exception class that captures stack trace at throw site on Windows.
// The stack trace is captured in the constructor, preserving the call stack
// at the point where the exception is created/thrown.
class StackTraceException : public std::runtime_error
{
public:
    StackTraceException(std::string const& what)
        : std::runtime_error(what.c_str())
    {
#ifdef _WIN32
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
