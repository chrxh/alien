#include <iostream>

#include "LoggingServiceImpl.h"

void LoggingServiceImpl::logMessage(char const* message) const
{
    std::cerr << message << std::endl;
}
