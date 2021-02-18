#pragma once
#include "Base/LoggingService.h"
#include "Definitions.h"

class FileLogger : public LoggingCallBack
{

public:
    FileLogger();
    virtual ~FileLogger() = default;

    void newLogMessage(std::string const& message) override;
};
