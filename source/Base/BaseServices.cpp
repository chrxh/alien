#include "ServiceLocator.h"
#include "LoggingServiceImpl.h"
#include "BaseServices.h"

BaseServices::BaseServices()
{
    static LoggingServiceImpl loggingServiceImpl;

    ServiceLocator::getInstance().registerService<LoggingService>(&loggingServiceImpl);
}
