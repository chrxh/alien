#include "LoggingServiceImpl.h"

#include <iostream>
#include <iomanip>
#include <ctime>
#include <sstream>

void LoggingServiceImpl::logMessage(std::string const& message) const
{
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);

    std::stringstream stream;
    stream << std::put_time(&tm, "%Y-%m-%d %H-%M-%S") << ": " << message;

    auto enrichedMessage = stream.str();
    for (auto const& callback : _callbacks) {
        callback->newLogMessage(enrichedMessage);
    }
}

void LoggingServiceImpl::registerCallBack(LoggingCallBack* callback)
{
    _callbacks.emplace_back(callback);
}
