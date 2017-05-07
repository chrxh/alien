#include "global/ServiceLocator.h"
#include "model/features/CellFeatureFactory.h"

#include "EnergyParticleImpl.h"
#include "CellImpl.h"
#include "CellClusterImpl.h"
#include "TokenImpl.h"
#include "EntityFactoryImpl.h"

namespace
{
    EntityFactoryImpl instance;
}

EntityFactoryImpl::EntityFactoryImpl ()
{
    ServiceLocator::getInstance().registerService<EntityFactory>(this);
}

CellCluster* EntityFactoryImpl::buildCellCluster (UnitContext* context) const
{
    return new CellClusterImpl(context);
}

CellCluster* EntityFactoryImpl::buildCellClusterFromForeignCells (QList< Cell* > cells
    , qreal angle, UnitContext* context) const
{
    return new CellClusterImpl(cells, angle, context);
}

Cell* EntityFactoryImpl::buildCell (UnitContext* context) const
{
    return new CellImpl(context);
}

Token* EntityFactoryImpl::buildToken (UnitContext* context) const
{
    return new TokenImpl(context);
}

Token * EntityFactoryImpl::buildTokenWithRandomData(UnitContext* context, qreal energy) const
{
	return new TokenImpl(context, energy, true);
}

EnergyParticle* EntityFactoryImpl::buildEnergyParticle(UnitContext* context) const
{
    return new EnergyParticleImpl(context);
}

CellCluster* EntityFactoryImpl::build(CellClusterDescription const& desc, UnitContext* context) const
{
	list<Cell*> cells;
	map<uint64_t, Cell*> cellsByIds;
	for (auto const &cellDesc : desc.cells) {
		auto cell = build(cellDesc, context);
		cells.push_back(cell);
		cellsByIds[cellDesc.id] = cell;
	}
	for (auto const &connection : desc.cellConnections) {
		uint64_t id1 = connection.first;
		uint64_t id2 = connection.second;
		Cell* cell1 = cellsByIds[id1];
		Cell* cell2 = cellsByIds[id2];
		cell1->newConnection(cell2);
	}
	return new CellClusterImpl(QList<Cell*>::fromStdList(cells), desc.angle, desc.pos, desc.angularVel, desc.vel, context);
}

Cell * EntityFactoryImpl::build(CellDescription const & desc, UnitContext * context) const
{
	CellFeatureFactory* featureFactory = ServiceLocator::getInstance().getService<CellFeatureFactory>();
	auto cell = new CellImpl(desc.energy, context, desc.maxConnections, desc.tokenAccessNumber, desc.relPos);
	cell->setFlagTokenBlocked(desc.tokenBlocked);
	cell->setMetadata(desc.metadata);
	featureFactory->addCellFunction(cell, desc.cellFunction.type, desc.cellFunction.data, context);
	featureFactory->addEnergyGuidance(cell, context);
	for (auto const& tokenDesc : desc.tokens) {
		cell->addToken(build(tokenDesc, context));
	}
	return cell;
}

Token * EntityFactoryImpl::build(TokenDescription const & desc, UnitContext * context) const
{
	return new TokenImpl(context, desc.energy, desc.data);
}

CellCluster * EntityFactoryImpl::build(CellClusterLightDescription const & desc, UnitContext * context) const
{
	return nullptr;
}

EnergyParticle* EntityFactoryImpl::build(EnergyParticleDescription const& desc, UnitContext* context) const
{
	auto particle = new EnergyParticleImpl(desc.energy, desc.pos, desc.vel, context);
	particle->setMetadata(desc.metadata);
	return particle;
}

EnergyParticle * EntityFactoryImpl::build(EnergyParticleLightDescription const & desc, UnitContext * context) const
{
	return new EnergyParticleImpl(desc.energy, desc.pos, desc.vel, context);
}

