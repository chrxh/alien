#include "StackTraceHelper.h"

#include "LoggingService.h"

namespace StackTraceHelper
{
    void logExceptionWithStackTrace(std::string const& exceptionMessage)
    {
        std::string logMessage = "Exception occurred: " + exceptionMessage;

#ifdef _WIN32
        std::string stackTrace = captureStackTrace();
        if (!stackTrace.empty()) {
            logMessage += "\nStack trace:\n" + stackTrace;
        }
#endif

        log(Priority::Important, logMessage);
    }
}
