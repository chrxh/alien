#include "LoggingService.h"

#include <algorithm>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>

void LoggingService::log(Priority priority, std::string const& message)
{
    std::lock_guard<std::mutex> lock(_mutex);

    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);

    std::stringstream stream;
    stream << std::put_time(&tm, "%Y-%m-%d %H-%M-%S") << ": " << message;

    auto enrichedMessage = stream.str();
    _messages.emplace_back(enrichedMessage);
    for (auto const& callback : _callbacks) {
        callback->newLogMessage(priority, enrichedMessage);
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
