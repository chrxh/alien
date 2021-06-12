#pragma once

#include <exception>
#include <string>

class BugReportException : public std::runtime_error
{
public:
    BugReportException(std::string const& what)
        : std::runtime_error(what.c_str())
    {}
};

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

class ParseErrorException : public std::runtime_error
{
public:
    ParseErrorException(std::string const& message)
        : std::runtime_error(message.c_str())
    {}
};
