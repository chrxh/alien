#include "Base/ServiceLocator.h"

#include "ModelCpuBuilderFacadeImpl.h"
#include "AccessPortFactoryImpl.h"
#include "ContextFactoryImpl.h"
#include "EntityFactoryImpl.h"
#include "CellFeatureFactoryImpl.h"

#include "ModelCpuServices.h"

ModelCpuServices::ModelCpuServices()
{
	static ModelCpuBuilderFacadeImpl modelBuilder;
	static AccessPortFactoryImpl accessPortFactory;
	static ContextFactoryImpl contextFactory;
	static EntityFactoryImpl entityFactory;
	static CellFeatureFactoryImpl featureFactory;

	ServiceLocator::getInstance().registerService<ModelCpuBuilderFacade>(&modelBuilder);
	ServiceLocator::getInstance().registerService<AccessPortFactory>(&accessPortFactory);
	ServiceLocator::getInstance().registerService<ContextFactory>(&contextFactory);
	ServiceLocator::getInstance().registerService<EntityFactory>(&entityFactory);
	ServiceLocator::getInstance().registerService<CellFeatureFactory>(&featureFactory);
}
