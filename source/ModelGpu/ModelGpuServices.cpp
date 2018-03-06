#include "Base/ServiceLocator.h"

#include "ModelGpu/_Impl/ModelGpuBuilderFacadeImpl.h"

#include "ModelGpuServices.h"

ModelGpuServices::ModelGpuServices()
{
	static ModelGpuBuilderFacadeImpl modelGpuBuilder;

	ServiceLocator::getInstance().registerService<ModelGpuBuilderFacade>(&modelGpuBuilder);
}
