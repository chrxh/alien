#include "FileLogger.h"

#include <Base/LoggingService.h>
#include <Base/Resources.h>

#include "Definitions.h"

_FileLogger::_FileLogger()
{
    LoggingService::get().registerCallBack(this);

    try {
        std::filesystem::remove(Const::LogFilename);
        _outfile.open(Const::LogFilename, std::ios_base::app);
    } catch (...) {
    }
}

_FileLogger::~_FileLogger()
{
    LoggingService::get().unregisterCallBack(this);
}

void _FileLogger::newLogMessage(Priority priority, std::string const& message)
{
    if (_outfile.is_open()) {
        _outfile << message << std::endl;
    }
}
