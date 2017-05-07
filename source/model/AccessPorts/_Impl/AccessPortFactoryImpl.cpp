#include "global/ServiceLocator.h"
#include "model/context/SimulationParameters.h"
#include "model/AccessPorts/Descriptions.h"
#include "model/AccessPorts/LightDescriptions.h"
#include "model/entities/EntityFactory.h"
#include "model/entities/Cell.h"
#include "model/entities/EnergyParticle.h"
#include "model/features/CellFeatureFactory.h"
#include "SimulationAccessImpl.h"
#include "AccessPortFactoryImpl.h"

namespace
{
	AccessPortFactoryImpl instance;
}

AccessPortFactoryImpl::AccessPortFactoryImpl()
{
	ServiceLocator::getInstance().registerService<AccessPortFactory>(this);
}

SimulationFullAccess * AccessPortFactoryImpl::buildSimulationFullAccess() const
{
	return new SimulationAccessImpl<DataDescription>();
}

SimulationLightAccess * AccessPortFactoryImpl::buildSimulationLightAccess() const
{
	return new SimulationAccessImpl<DataLightDescription>();
}

CellCluster* AccessPortFactoryImpl::buildFromDescription(CellClusterDescription const& desc, UnitContext* context) const
{
	EntityFactory* entityFactory = ServiceLocator::getInstance().getService<EntityFactory>();
	CellFeatureFactory* featureFactory = ServiceLocator::getInstance().getService<CellFeatureFactory>();

	list<Cell*> cells;
	for (auto const& cellDesc : desc.cells) {
		auto cell = entityFactory->buildCell(cellDesc.energy, context, cellDesc.maxConnections, cellDesc.tokenAccessNumber, cellDesc.relPos);
		cell->setMetadata(cellDesc.metadata);
		featureFactory->addCellFunction(cell, cellDesc.cellFunction.type, cellDesc.cellFunction.data, context);
		featureFactory->addEnergyGuidance(cell, context);
		cells.push_back(cell);
	}
	return entityFactory->buildCellCluster(QList<Cell*>::fromStdList(cells), desc.angle, desc.pos, desc.angularVel, desc.vel, context);
}

CellCluster * AccessPortFactoryImpl::buildFromDescription(CellClusterLightDescription const & desc, UnitContext * context) const
{
	return nullptr;
}

EnergyParticle* AccessPortFactoryImpl::buildFromDescription(EnergyParticleDescription const& desc, UnitContext* context) const
{
	EntityFactory* entityFactory = ServiceLocator::getInstance().getService<EntityFactory>();
	auto particle = entityFactory->buildEnergyParticle(desc.energy, desc.pos, desc.vel, context);
	particle->setMetadata(desc.metadata);
	return particle;
}

EnergyParticle * AccessPortFactoryImpl::buildFromDescription(EnergyParticleLightDescription const & desc, UnitContext * context) const
{
	EntityFactory* entityFactory = ServiceLocator::getInstance().getService<EntityFactory>();
	return entityFactory->buildEnergyParticle(desc.energy, desc.pos, desc.vel, context);
}

