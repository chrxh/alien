#include "Base/ServiceLocator.h"

#include "ModelBasicBuilderFacadeImpl.h"
#include "DescriptionFactoryImpl.h"

#include "ModelBasicServices.h"

ModelBasicServices::ModelBasicServices()
{
	static ModelBasicBuilderFacadeImpl modelBuilder;
    static DescriptionFactoryImpl descriptionFactory;

	ServiceLocator::getInstance().registerService<ModelBasicBuilderFacade>(&modelBuilder);
    ServiceLocator::getInstance().registerService<DescriptionFactory>(&descriptionFactory);
}
