#pragma once

#include <Base/LoggingService.h>

#include "Definitions.h"

class _GuiLogger : public LoggingCallBack
{
public:
    _GuiLogger();
    ~_GuiLogger() override;

    std::vector<LogMessage> const& getMessages(Priority minPriority) const;

private:
    void newLogMessage(LogMessage const& message) override;

    std::vector<LogMessage> _allLogMessages;
    std::vector<LogMessage> _importantLogMessages;
};
