#include "FileLogger.h"

#include "Base/LoggingService.h"
#include "Base/Resources.h"

#include "Definitions.h"

_FileLogger::_FileLogger()
{
    LoggingService::get().registerCallBack(this);

    std::remove(Const::LogFilename.c_str());
    _outfile.open(Const::LogFilename, std::ios_base::app);
}

_FileLogger::~_FileLogger()
{
    LoggingService::get().unregisterCallBack(this);
}

void _FileLogger::newLogMessage(Priority priority, std::string const& message)
{
    _outfile << message << std::endl;
}
