#include "FileLogger.h"

#include <Base/Interface/LoggingService.h>
#include <Base/Interface/Resources.h>

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

void _FileLogger::newLogMessage(LogMessage const& message)
{
    if (_outfile.is_open()) {
        _outfile << LoggingService::format(message) << std::endl;
    }
}
