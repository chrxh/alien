#pragma once
#include <sstream>

#include "Base/LoggingService.h"
#include "Definitions.h"

class BugReportLogger : public LoggingCallBack
{

public:
    BugReportLogger();
    virtual ~BugReportLogger();

    void newLogMessage(Priority priority, std::string const& message) override;

    std::string getFullProtocol() const;

private:
    std::stringstream _stream;
};
