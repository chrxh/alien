#include "Base/ServiceLocator.h"

#include "ModelGpuBuilderFacadeImpl.h"
#include "ModelGpuServices.h"

ModelGpuServices::ModelGpuServices()
{
	static ModelGpuBuilderFacadeImpl modelGpuBuilder;

	ServiceLocator::getInstance().registerService<ModelGpuBuilderFacade>(&modelGpuBuilder);
}
