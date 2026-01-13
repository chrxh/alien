#include "StackTraceHelper.h"

#include "LoggingService.h"

namespace StackTraceHelper
{
    void logException(StackTraceException const& exception)
    {
        std::string logMessage = "Exception occurred: " + std::string(exception.what());

#ifdef _WIN32
        std::string const& stackTrace = exception.getStackTrace();
        if (!stackTrace.empty()) {
            logMessage += "\nStack trace:\n" + stackTrace;
        }
#endif

        log(Priority::Important, logMessage);
    }

    void logException(std::exception const& exception)
    {
        std::string logMessage = "Exception occurred: " + std::string(exception.what());
        log(Priority::Important, logMessage);
    }

    void logUnknownException()
    {
        log(Priority::Important, "Unknown exception occurred");
    }
}
