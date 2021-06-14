#include <QMetaType>

#include "ServiceLocator.h"
#include "LoggingServiceImpl.h"
#include "GlobalFactoryImpl.h"
#include "BaseServices.h"

BaseServices::BaseServices()
{
    static LoggingServiceImpl loggingServiceImpl;
    static GlobalFactoryImpl globalFactoryImpl;

    ServiceLocator::getInstance().registerService<LoggingService>(&loggingServiceImpl);
	ServiceLocator::getInstance().registerService<GlobalFactory>(&globalFactoryImpl);
}
