#pragma once

#include <fstream>

#include "Base/LoggingService.h"
#include "Definitions.h"

class _FileLogger : public LoggingCallBack
{

public:
    _FileLogger();
    ~_FileLogger() override;

    void newLogMessage(Priority priority, std::string const& message) override;

private:
    std::ofstream _outfile;
};
