#pragma once

#include <fstream>

#include "Base/LoggingService.h"
#include "Definitions.h"

class FileLogger : public LoggingCallBack
{

public:
    FileLogger();
    virtual ~FileLogger() = default;

    void newLogMessage(Priority priority, std::string const& message) override;

private:
    std::ofstream _outfile;
};
