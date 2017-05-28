#include "Base/ServiceLocator.h"

#include "Model/_Impl/ModelBuilderFacadeImpl.h"
#include "Model/_Impl/SerializationFacadeImpl.h"
#include "Model/AccessPorts/_Impl/AccessPortFactoryImpl.h"
#include "Model/Context/_Impl/ContextFactoryImpl.h"
#include "Model/Entities/_Impl/EntityFactoryImpl.h"
#include "Model/Features/_Impl/CellFeatureFactoryImpl.h"

#include "modelservices.h"

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
