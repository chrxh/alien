#include "WebServices.h"

#include <QMetaType>

#include "Base/ServiceLocator.h"

#include "WebBuilderFacadeImpl.h"

WebServices::WebServices()
{
    static WebBuilderFacadeImpl webBuilder;

    ServiceLocator::getInstance().registerService<WebBuilderFacade>(&webBuilder);
}
