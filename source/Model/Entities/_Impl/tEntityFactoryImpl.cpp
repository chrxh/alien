#include "Base/ServiceLocator.h"
#include "model/Features/CellFeatureFactory.h"

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

CellCluster* EntityFactoryImpl::build(CellClusterDescription const& desc, UnitContext* context) const
{
	list<Cell*> cells;
	map<uint64_t, Cell*> cellsByIds;
	for (auto const &cellDesc : desc.cells) {
		auto cell = build(cellDesc.getValue(), context);
		cells.push_back(cell);
		cellsByIds[cellDesc.getValue().id] = cell;
	}
	for (auto const &connection : desc.cellConnections) {
		auto const& conValue = connection.getValue();
		uint64_t id1 = conValue.first;
		uint64_t id2 = conValue.second;
		Cell* cell1 = cellsByIds[id1];
		Cell* cell2 = cellsByIds[id2];
		cell1->newConnection(cell2);
	}
	return new CellClusterImpl(QList<Cell*>::fromStdList(cells), desc.angle.getValueOr(0.0), desc.pos.getValue()
		, desc.angularVel.getValueOr(0.0), desc.vel.getValueOr(QVector2D()), context);
}

Cell * EntityFactoryImpl::build(CellDescription const & desc, UnitContext * context) const
{
	CellFeatureFactory* featureFactory = ServiceLocator::getInstance().getService<CellFeatureFactory>();
	auto const& energy = desc.energy.getValue();
	auto const& maxConnections = desc.maxConnections.getValueOr(0);
	auto const& tokenAccessNumber = desc.tokenAccessNumber.getValueOr(0);
	auto const& relPos = desc.pos.getValueOr({ 0.0, 0.0 });
	auto cell = new CellImpl(energy, context, maxConnections, tokenAccessNumber, relPos);
	cell->setFlagTokenBlocked(desc.tokenBlocked.getValueOr(false));
	cell->setMetadata(desc.metadata.getValueOr(CellMetadata()));

	auto const& cellFunction = desc.cellFunction.getValueOr(CellFunctionDescription());
	featureFactory->addCellFunction(cell, cellFunction.type, cellFunction.data, context);
	featureFactory->addEnergyGuidance(cell, context);
	auto const& tokensDesc = desc.tokens.getValueOr(vector<TokenDescription>());
	for (auto const& tokenDesc : tokensDesc) {
		cell->addToken(build(tokenDesc, context));
	}
	return cell;
}

Token * EntityFactoryImpl::build(TokenDescription const & desc, UnitContext * context) const
{
	auto const& data = desc.data;
	auto const& energy = desc.energy;
	return new TokenImpl(context, energy, data);
}

EnergyParticle* EntityFactoryImpl::build(EnergyParticleDescription const& desc, UnitContext* context) const
{
	auto const& pos = desc.pos.getValue();
	auto const&vel = desc.vel.getValueOr({ 0.0, 0.0 });
	auto particle = new EnergyParticleImpl(desc.energy.getValue(), pos, vel, context);
	auto const& metadata = desc.metadata.getValueOr(EnergyParticleMetadata());
	particle->setMetadata(metadata);
	return particle;
}


