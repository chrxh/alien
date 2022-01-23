#include "LoggingService.h"

#include <iostream>
#include <iomanip>
#include <ctime>
#include <sstream>
#include <algorithm>

LoggingService& LoggingService::getInstance()
{
    static LoggingService instance;
    return instance;
}

void LoggingService::log(Priority priority, std::string const& message)
{
    std::lock_guard<std::mutex> lock(_mutex);

    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);

    std::stringstream stream;
    stream << std::put_time(&tm, "%Y-%m-%d %H-%M-%S") << ": " << message;

    auto enrichedMessage = stream.str();
    for (auto const& callback : _callbacks) {
        callback->newLogMessage(priority, enrichedMessage);
    }
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
