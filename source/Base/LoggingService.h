#pragma once

#include <string>
#include <vector>
#include <mutex>

#include "Singleton.h"

enum class Priority
{
    Unimportant,
    Important,
};

class LoggingCallBack
{
public:
    virtual ~LoggingCallBack() = default;
    virtual void newLogMessage(Priority priority, std::string const& message) = 0;
};

class LoggingService
{
    MAKE_SINGLETON(LoggingService);

public:
    void log(Priority priority, std::string const& message);

    void registerCallBack(LoggingCallBack* callback);
    void unregisterCallBack(LoggingCallBack* callback);

private:
    std::vector<LoggingCallBack*> _callbacks;
    std::mutex _mutex;
};

inline void log(Priority priority, std::string const& message)
{
    LoggingService::get().log(priority, message);
}