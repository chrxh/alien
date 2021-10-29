#pragma once

#include <fstream>

#include "Base/LoggingService.h"
#include "Definitions.h"

class _FileLogger : public LoggingCallBack
{

public:
    _FileLogger();
    virtual ~_FileLogger();

    void newLogMessage(Priority priority, std::string const& message) override;

private:
    std::ofstream _outfile;
};
