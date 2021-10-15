#include "SimpleLogger.h"

#include "Base/ServiceLocator.h"
#include "Base/LoggingService.h"

_SimpleLogger::_SimpleLogger()
{
    auto loggingService = ServiceLocator::getInstance().getService<LoggingService>();
    loggingService->registerCallBack(this);
}

_SimpleLogger::~_SimpleLogger()
{
    auto loggingService = ServiceLocator::getInstance().getService<LoggingService>();
    loggingService->unregisterCallBack(this);
}

std::vector<std::string> const& _SimpleLogger::getMessages(Priority minPriority) const
{
    if (Priority::Important == minPriority) {
        return _importantLogMessages;
    }
    return _allLogMessages;
}

void _SimpleLogger::newLogMessage(Priority priority, std::string const& message)
{
    _allLogMessages.emplace_back(message);
    if (Priority::Important == priority) {
        _importantLogMessages.emplace_back(message);
    }
}
