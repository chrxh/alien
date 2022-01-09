#include "SimpleLogger.h"

#include "Base/LoggingService.h"

_SimpleLogger::_SimpleLogger()
{
    LoggingService::getInstance().registerCallBack(this);
}

_SimpleLogger::~_SimpleLogger()
{
    LoggingService::getInstance().unregisterCallBack(this);
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
