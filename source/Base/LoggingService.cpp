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

    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    auto offset = formatTime(*std::localtime(&t), "%z");
    if (_timezoneOffset != offset) {
        _timezoneOffset = offset;
        addMessage(LogMessage{.time = now, .priority = Priority::Important, .text = "log timezone is " + offset});
    }
    addMessage(LogMessage{.time = now, .priority = priority, .text = message});
}

void LoggingService::addMessage(LogMessage const& message)
{
    _messages.emplace_back(message);
    for (auto const& callback : _callbacks) {
        callback->newLogMessage(message);
    }
}

std::string LoggingService::getLogString() const
{
    std::lock_guard<std::mutex> lock(_mutex);

    std::stringstream stream;
    for (auto const& message : _messages) {
        stream << format(message) << std::endl;
    }
    return stream.str();
}

std::string LoggingService::format(LogMessage const& message)
{
    auto t = std::chrono::system_clock::to_time_t(message.time);
    return formatTime(*std::localtime(&t), "%Y-%m-%d %H-%M-%S") + ": " + message.text;
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
