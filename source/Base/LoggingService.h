#pragma once

#include <string>

enum class Priority
{
    Unimportant,
    Important,
};

class LoggingCallBack
{
public:
    virtual void newLogMessage(Priority priority, std::string const& message) = 0;
};

class LoggingService
{
public:
    virtual ~LoggingService() = default;

    virtual void logMessage(Priority priority, std::string const& message) const = 0;

    virtual void registerCallBack(LoggingCallBack* callback) = 0;
};
