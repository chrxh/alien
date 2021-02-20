#include "LoggingServiceImpl.h"

#include <iostream>
#include <iomanip>
#include <ctime>
#include <sstream>

void LoggingServiceImpl::logMessage(Priority priority, std::string const& message)
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

void LoggingServiceImpl::registerCallBack(LoggingCallBack* callback)
{
    _callbacks.emplace_back(callback);
}
