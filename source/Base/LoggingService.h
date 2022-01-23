#pragma once

#include <string>
#include <vector>
#include <mutex>

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
    static LoggingService& getInstance();

    void log(Priority priority, std::string const& message);

    void registerCallBack(LoggingCallBack* callback);
    void unregisterCallBack(LoggingCallBack* callback);

private:
    std::vector<LoggingCallBack*> _callbacks;
    std::mutex _mutex;
};

inline void log(Priority priority, std::string const& message)
{
    LoggingService::getInstance().log(priority, message);
}