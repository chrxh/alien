#include "Base/ServiceLocator.h"
#include "Base/NumberGenerator.h"
#include "Model/Local/CellFeatureFactory.h"

#include "Particle.h"
#include "Cell.h"
#include "Cluster.h"
#include "TokenImpl.h"
#include "EntityFactoryImpl.h"

Cluster* EntityFactoryImpl::build(ClusterDescription const& desc, UnitContext* context) const
{
	uint64_t id = desc.id == 0 ? context->getNumberGenerator()->getTag() : desc.id;
	auto result = new Cluster(QList<Cell*>(), id, desc.angle.get_value_or(0.0), desc.pos.get()
		, desc.angularVel.get_value_or(0.0), desc.vel.get_value_or(QVector2D()), context);

	if(desc.metadata) {
		result->setMetadata(*desc.metadata);
	}

	if (desc.cells) {
		map<uint64_t, Cell*> cellsByIds;
		for (auto const &cellDesc : *desc.cells) {
			auto cell = build(cellDesc, result, context);
			cellsByIds[cellDesc.id] = cell;
			if (desc.id == 0) {
				cell->setId(context->getNumberGenerator()->getTag());	//generate id for cell if cluster has invalid id
			}
		}

		for (auto const &cellDesc : *desc.cells) {
			if (!cellDesc.connectingCells) {
				continue;
			}
			for (uint64_t connectingCellId : *cellDesc.connectingCells) {
				Cell* cell1 = cellsByIds[cellDesc.id];
				Cell* cell2 = cellsByIds[connectingCellId];
				if (!cell1->isConnectedTo(cell2)) {
					cell1->newConnection(cell2);
				}
			}
		}
	}
	result->updateInternals();
	return result;
}

Cell * EntityFactoryImpl::build(CellDescription const& cellDesc, Cluster* cluster, UnitContext* context) const
{
	auto const& energy = *cellDesc.energy;
	auto const& maxConnections = cellDesc.maxConnections.get_value_or(0);
	auto const& tokenAccessNumber = cellDesc.tokenBranchNumber.get_value_or(0);
	uint64_t id = cellDesc.id == 0 ? context->getNumberGenerator()->getTag() : cellDesc.id;
	auto cell = new Cell(id, energy, context, maxConnections, tokenAccessNumber);
	if (cellDesc.tokenBlocked) {
		cell->setFlagTokenBlocked(*cellDesc.tokenBlocked);
	}
	if (cellDesc.metadata) {
		cell->setMetadata(*cellDesc.metadata);
	}
	auto const& cellFunction = cellDesc.cellFeature.get_value_or(CellFeatureDescription());

	CellFeatureFactory* featureFactory = ServiceLocator::getInstance().getService<CellFeatureFactory>();
	auto features = featureFactory->build(cellFunction, context);
	cell->registerFeatures(features);
	if (cellDesc.tokens) {
		for (auto const& tokenDesc : *cellDesc.tokens) {
			cell->addToken(build(tokenDesc, context), Cell::ActivateToken::Now, Cell::UpdateTokenBranchNumber::No);
		}
	}
	cluster->addCell(cell, cellDesc.pos.get_value_or({ 0.0, 0.0 }), Cluster::UpdateInternals::No);
	return cell;
}

Token * EntityFactoryImpl::build(TokenDescription const & desc, UnitContext * context) const
{
	auto const& data = desc.data.get_value_or(QByteArray());
	auto const& energy = desc.energy.get_value_or(0.0);
	return new TokenImpl(context, energy, data);
}

Particle* EntityFactoryImpl::build(ParticleDescription const& desc, UnitContext* context) const
{
	auto const& pos = *desc.pos;
	auto const& vel = desc.vel.get_value_or({ 0.0, 0.0 });
	uint64_t id = desc.id == 0 ? context->getNumberGenerator()->getTag() : desc.id;
	auto particle = new Particle(id, *desc.energy, pos, vel, context);
	auto const& metadata = desc.metadata.get_value_or(ParticleMetadata());
	particle->setMetadata(metadata);
	return particle;
}


