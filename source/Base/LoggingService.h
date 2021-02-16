#pragma once

#include <string>

class LoggingCallBack
{
public:
    virtual void newLogMessage(std::string const& message) = 0;
};

class LoggingService
{
public:
    virtual ~LoggingService() = default;

    virtual void logMessage(std::string const& message) const = 0;

    virtual void registerCallBack(LoggingCallBack* callback) = 0;
};
