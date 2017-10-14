#include "Base/ServiceLocator.h"

#include "Model/Impl/ModelBuilderFacadeImpl.h"
#include "Model/Impl/SerializationFacadeImpl.h"
#include "Model/Impl/AccessPortFactoryImpl.h"
#include "Model/Impl/ContextFactoryImpl.h"
#include "Model/Impl/EntityFactoryImpl.h"
#include "Model/Impl/CellFeatureFactoryImpl.h"

#include "ModelServices.h"

ModelServices::ModelServices()
{
	static ModelBuilderFacadeImpl modelBuilder;
	static SerializationFacadeImpl serializationFacade;
	static AccessPortFactoryImpl accessPortFactory;
	static ContextFactoryImpl contextFactory;
	static EntityFactoryImpl entityFactory;
	static CellFeatureFactoryImpl featureFactory;

	ServiceLocator::getInstance().registerService<ModelBuilderFacade>(&modelBuilder);
	ServiceLocator::getInstance().registerService<SerializationFacade>(&serializationFacade);
	ServiceLocator::getInstance().registerService<AccessPortFactory>(&accessPortFactory);
	ServiceLocator::getInstance().registerService<ContextFactory>(&contextFactory);
	ServiceLocator::getInstance().registerService<EntityFactory>(&entityFactory);
	ServiceLocator::getInstance().registerService<CellFeatureFactory>(&featureFactory);
}
