#include <QMetaType>

#include "Base/ServiceLocator.h"

#include "EngineGpuBuilderFacadeImpl.h"
#include "EngineGpuServices.h"

EngineGpuServices::EngineGpuServices()
{
	static EngineGpuBuilderFacadeImpl EngineGpuBuilder;

	ServiceLocator::getInstance().registerService<EngineGpuBuilderFacade>(&EngineGpuBuilder);
	
}
