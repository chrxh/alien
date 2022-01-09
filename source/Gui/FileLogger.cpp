#include "FileLogger.h"

#include "Base/LoggingService.h"

#include "Definitions.h"
#include "Resources.h"

_FileLogger::_FileLogger()
{
    LoggingService::getInstance().registerCallBack(this);

    std::remove(Const::LogFilename);
    _outfile.open(Const::LogFilename, std::ios_base::app);
}

_FileLogger::~_FileLogger()
{
    LoggingService::getInstance().unregisterCallBack(this);
}

void _FileLogger::newLogMessage(Priority priority, std::string const& message)
{
    _outfile << message << std::endl;
}
