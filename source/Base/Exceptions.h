#pragma once

#include <exception>
#include <stdexcept>
#include <string>

// Function to capture stack trace at call site.
// Implemented in Exceptions.cpp which is compiled with C++23.
// This allows stack trace capture even when the header is included in CUDA code.
std::string captureStackTrace();

// Base exception class that captures stack trace at throw site.
// The stack trace is captured in the constructor, preserving the call stack
// at the point where the exception is created/thrown.
// Works on Windows with MSVC and C++23.
class StackTraceException : public std::runtime_error
{
public:
    StackTraceException(std::string const& what)
        : std::runtime_error(what)
        , _stackTrace(captureStackTrace())
    {}

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
