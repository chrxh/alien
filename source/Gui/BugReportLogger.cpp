#include "BugReportLogger.h"

#include "Base/LoggingService.h"
#include "Base/ServiceLocator.h"

BugReportLogger::BugReportLogger()
{
    auto loggingService = ServiceLocator::getInstance().getService<LoggingService>();
    loggingService->registerCallBack(this);
}

BugReportLogger::~BugReportLogger()
{
    auto loggingService = ServiceLocator::getInstance().getService<LoggingService>();
    loggingService->unregisterCallBack(this);
}

void BugReportLogger::newLogMessage(Priority priority, std::string const& message)
{
    _stream << message << std::endl;
}

std::string BugReportLogger::getFullProtocol() const
{
    return _stream.str();
}
