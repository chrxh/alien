#pragma once

class LoggingService
{
public:
    virtual ~LoggingService() = default;

    virtual void logMessage(char const* message) const = 0;
};
