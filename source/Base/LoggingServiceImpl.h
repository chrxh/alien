#pragma once
#include <vector>

#include "LoggingService.h"

class LoggingServiceImpl : public LoggingService
{
public:
    virtual ~LoggingServiceImpl() = default;

    void logMessage(std::string const& message) const override;

    void registerCallBack(LoggingCallBack* callback) override;

private:
    std::vector<LoggingCallBack*> _callbacks;
};
