#pragma once

#include <exception>
#include <stdexcept>
#include <string>

class InitialCheckException : public std::runtime_error
{
public:
    InitialCheckException(std::string const& what)
        : std::runtime_error(what.c_str())
    {}
};

class CudaMemoryAllocationException : public std::runtime_error
{
public:
    CudaMemoryAllocationException(std::string const& what)
        : std::runtime_error(what.c_str())
    {}
};
