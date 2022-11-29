#pragma once

#include <exception>
#include <stdexcept>
#include <string>

class SpecificCudaException : public std::runtime_error
{
public:
    SpecificCudaException(std::string const& what)
        : std::runtime_error(what.c_str())
    {}
};

class SystemRequirementNotMetException : public std::runtime_error
{
public:
    SystemRequirementNotMetException(std::string const& what)
        : std::runtime_error(what.c_str())
    {}
};
