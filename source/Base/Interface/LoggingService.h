#pragma once

#include <chrono>
#include <mutex>
#include <string>
#include <vector>

#include "Singleton.h"

enum class Priority
{
    Unimportant,
    Important,
};

struct LogMessage
{
    std::chrono::system_clock::time_point time;
    Priority priority = Priority::Important;
    std::string text;
};

class LoggingCallBack
{
public:
    virtual ~LoggingCallBack() = default;
    virtual void newLogMessage(LogMessage const& message) = 0;
};

class LoggingService
{
    MAKE_SINGLETON(LoggingService);

public:
    void log(Priority priority, std::string const& message);
    std::string getLogString() const;

    static std::string format(LogMessage const& message);

    void registerCallBack(LoggingCallBack* callback);
    void unregisterCallBack(LoggingCallBack* callback);

private:
    void addMessage(LogMessage const& message);

    std::vector<LoggingCallBack*> _callbacks;
    std::vector<LogMessage> _messages;
    std::string _timezoneOffset;
    mutable std::mutex _mutex;
};

inline void log(Priority priority, std::string const& message)
{
    LoggingService::get().log(priority, message);
}
