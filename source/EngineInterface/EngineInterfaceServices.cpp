#include "Base/ServiceLocator.h"

#include "EngineInterfaceBuilderFacadeImpl.h"
#include "DescriptionFactoryImpl.h"

#include "EngineInterfaceServices.h"

EngineInterfaceServices::EngineInterfaceServices()
{
	static EngineInterfaceBuilderFacadeImpl modelBuilder;
    static DescriptionFactoryImpl descriptionFactory;

	ServiceLocator::getInstance().registerService<EngineInterfaceBuilderFacade>(&modelBuilder);
    ServiceLocator::getInstance().registerService<DescriptionFactory>(&descriptionFactory);
}
