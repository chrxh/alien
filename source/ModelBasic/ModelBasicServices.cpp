#include "Base/ServiceLocator.h"

#include "ModelBasicBuilderFacadeImpl.h"

#include "ModelBasicServices.h"

ModelBasicServices::ModelBasicServices()
{
	static ModelBasicBuilderFacadeImpl modelBuilder;

	ServiceLocator::getInstance().registerService<ModelBasicBuilderFacade>(&modelBuilder);
}
