#include "FileLogger.h"

#include <fstream>

#include "Base/ServiceLocator.h"
#include "Base/LoggingService.h"

namespace
{
    auto const LogFilename = "log.txt";
}

FileLogger::FileLogger()
{
    auto loggingService = ServiceLocator::getInstance().getService<LoggingService>();
    loggingService->registerCallBack(this);

    std::remove(LogFilename);
}

void FileLogger::newLogMessage(std::string const& message)
{
    std::ofstream outfile;

    outfile.open(LogFilename, std::ios_base::app);
    outfile << message << std::endl;
}
