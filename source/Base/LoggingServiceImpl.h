#pragma once
#include <vector>
#include <mutex>

#include "LoggingService.h"

class LoggingServiceImpl : public LoggingService
{
public:
    virtual ~LoggingServiceImpl() = default;

    void logMessage(Priority priority, std::string const& message) override;

    void registerCallBack(LoggingCallBack* callback) override;

private:
    std::vector<LoggingCallBack*> _callbacks;
    std::mutex _mutex;
};
