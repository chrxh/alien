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
		if (!cellD.connectingCells) {
			continue;
		}
		for (uint64_t connectingCellId : *cellD.connectingCells) {
			Cell* cell1 = cellsByIds[cellD.id];
			Cell* cell2 = cellsByIds[connectingCellId];
			if (!cell1->isConnectedTo(cell2)) {
				cell1->newConnection(cell2);
			}
		}
	}

	return new ClusterImpl(QList<Cell*>::fromStdList(cells), desc.angle.get_value_or(0.0), desc.pos.get()
		, desc.angularVel.get_value_or(0.0), desc.vel.get_value_or(QVector2D()), context);
}

Cell * EntityFactoryImpl::build(CellChangeDescription const & desc, UnitContext * context) const
{
	CellFeatureFactory* featureFactory = ServiceLocator::getInstance().getService<CellFeatureFactory>();
	auto const& energy = *desc.energy;
	auto const& maxConnections = desc.maxConnections.get_value_or(0);
	auto const& tokenAccessNumber = desc.tokenBranchNumber.get_value_or(0);
	auto const& relPos = desc.pos.get_value_or({ 0.0, 0.0 });
	auto cell = new CellImpl(energy, context, maxConnections, tokenAccessNumber, relPos);
	cell->setFlagTokenBlocked(desc.tokenBlocked.get_value_or(false));
	cell->setMetadata(desc.metadata.get_value_or(CellMetadata()));

	auto const& cellFunction = desc.cellFunction.get_value_or(CellFunctionDescription());
	featureFactory->addCellFunction(cell, cellFunction.type, cellFunction.data, context);
	featureFactory->addEnergyGuidance(cell, context);
	auto const& tokensDesc = desc.tokens.get_value_or(vector<TokenDescription>());
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
	auto const& pos = *desc.pos;
	auto const&vel = desc.vel.get_value_or({ 0.0, 0.0 });
	auto particle = new ParticleImpl(*desc.energy, pos, vel, context);
	auto const& metadata = desc.metadata.get_value_or(EnergyParticleMetadata());
	particle->setMetadata(metadata);
	return particle;
}


