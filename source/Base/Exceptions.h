#pragma once

#include <exception>
#include <string>

class BugReportException : public std::exception
{
public:
    BugReportException(std::string const& what)
        : std::exception(what.c_str())
    {}
};

class SystemRequirementNotMetException : public std::exception
{
public:
    SystemRequirementNotMetException(std::string const& what)
        : std::exception(what.c_str())
    {}
};

class ParseErrorException : public std::exception
{
public:
    ParseErrorException(std::string const& message)
        : std::exception(message.c_str())
    {}
};