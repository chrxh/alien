#include "Base/ServiceLocator.h"

#include "Model/Features/CellFeatureFactory.h"

#include "ParticleImpl.h"
#include "CellImpl.h"
#include "ClusterImpl.h"
#include "TokenImpl.h"
#include "EntityFactoryImpl.h"

Cluster* EntityFactoryImpl::build(ClusterChangeDescription const& desc, UnitContext* context) const
{
	list<Cell*> cells;
	map<uint64_t, Cell*> cellsByIds;
	for (auto const &cellT : desc.cells) {
		auto cell = build(cellT.getValue(), context);
		cells.push_back(cell);
		cellsByIds[cellT.getValue().id] = cell;
	}

	for (auto const &cellT : desc.cells) {
		auto cellD = cellT.getValue();
		if (!cellD.connectingCells.isInitialized()) {
			continue;
		}
		for (uint64_t connectingCellId : cellD.connectingCells.getValue()) {
			Cell* cell1 = cellsByIds[cellD.id];
			Cell* cell2 = cellsByIds[connectingCellId];
			if (!cell1->isConnectedTo(cell2)) {
				cell1->newConnection(cell2);
			}
		}
	}

	return new ClusterImpl(QList<Cell*>::fromStdList(cells), desc.angle.getValueOr(0.0), desc.pos.getValue()
		, desc.angularVel.getValueOr(0.0), desc.vel.getValueOr(QVector2D()), context);
}

Cell * EntityFactoryImpl::build(CellChangeDescription const & desc, UnitContext * context) const
{
	CellFeatureFactory* featureFactory = ServiceLocator::getInstance().getService<CellFeatureFactory>();
	auto const& energy = desc.energy.getValue();
	auto const& maxConnections = desc.maxConnections.getValueOr(0);
	auto const& tokenAccessNumber = desc.tokenBranchNumber.getValueOr(0);
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

Particle* EntityFactoryImpl::build(ParticleChangeDescription const& desc, UnitContext* context) const
{
	auto const& pos = desc.pos.getValue();
	auto const&vel = desc.vel.getValueOr({ 0.0, 0.0 });
	auto particle = new ParticleImpl(desc.energy.getValue(), pos, vel, context);
	auto const& metadata = desc.metadata.getValueOr(EnergyParticleMetadata());
	particle->setMetadata(metadata);
	return particle;
}


