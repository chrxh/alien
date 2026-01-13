#pragma once

#include <string>

#include "Exceptions.h"

namespace StackTraceHelper
{
    // Logs the exception message along with its stack trace to the logging service.
    // For StackTraceException types, the stack trace captured at throw time is logged.
    // For other exceptions, only the message is logged.
    void logException(StackTraceException const& exception);
    void logException(std::exception const& exception);
    void logUnknownException();
}
