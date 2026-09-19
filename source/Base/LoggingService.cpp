#include "LoggingService.h"

#include <algorithm>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace
{
    std::string formatTime(std::tm const& time, char const* format)
    {
        std::stringstream stream;
        stream << std::put_time(&time, format);
        return stream.str();
    }
}

void LoggingService::log(Priority priority, std::string const& message)
{
    std::lock_guard<std::mutex> lock(_mutex);

    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);

    auto timestamp = formatTime(tm, "%Y-%m-%d %H-%M-%S");
    auto offset = formatTime(tm, "%z");
    if (_timezoneOffset != offset) {
        _timezoneOffset = offset;
        addMessage(Priority::Important, timestamp + ": log timezone is " + offset);
    }
    addMessage(priority, timestamp + ": " + message);
}

void LoggingService::addMessage(Priority priority, std::string const& message)
{
    _messages.emplace_back(message);
    for (auto const& callback : _callbacks) {
        callback->newLogMessage(priority, message);
    }
}

std::string LoggingService::getLogString() const
{
    std::lock_guard<std::mutex> lock(_mutex);

    std::stringstream stream;
    for (auto const& message : _messages) {
        stream << message << std::endl;
    }
    return stream.str();
}

void LoggingService::registerCallBack(LoggingCallBack* callback)
{
    _callbacks.emplace_back(callback);
}

void LoggingService::unregisterCallBack(LoggingCallBack* callback)
{
    auto end = std::remove_if(_callbacks.begin(), _callbacks.end(), [&](auto const& callback_) { return callback_ == callback; });

    _callbacks.erase(end, _callbacks.end());
}
