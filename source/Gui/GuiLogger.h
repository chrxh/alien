#pragma once

#include "Base/LoggingService.h"
#include "Definitions.h"

class _GuiLogger : public LoggingCallBack
{
public:
    _GuiLogger();
    virtual ~_GuiLogger();

    std::vector<std::string> const& getMessages(Priority minPriority) const;

private:

    void newLogMessage(Priority priority, std::string const& message) override;

    std::vector<std::string> _allLogMessages;
    std::vector<std::string> _importantLogMessages;
};
