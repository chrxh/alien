#include "GuiLogger.h"

#include "Base/LoggingService.h"

_GuiLogger::_GuiLogger()
{
    LoggingService::get().registerCallBack(this);
}

_GuiLogger::~_GuiLogger()
{
    LoggingService::get().unregisterCallBack(this);
}

std::vector<std::string> const& _GuiLogger::getMessages(Priority minPriority) const
{
    if (Priority::Important == minPriority) {
        return _importantLogMessages;
    }
    return _allLogMessages;
}

void _GuiLogger::newLogMessage(Priority priority, std::string const& message)
{
    _allLogMessages.emplace_back(message);
    if (Priority::Important == priority) {
        _importantLogMessages.emplace_back(message);
    }
}
