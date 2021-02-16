#pragma once

#include "LoggingService.h"

class LoggingServiceImpl : public LoggingService
{
public:
    virtual ~LoggingServiceImpl() = default;

    void logMessage(char const* message) const override;
};
