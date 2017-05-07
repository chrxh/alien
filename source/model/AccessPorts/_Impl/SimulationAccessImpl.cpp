#include "global/ServiceLocator.h"
#include "model/context/SimulationContext.h"
#include "model/context/UnitContext.h"
#include "model/context/UnitThreadController.h"
#include "model/context/UnitGrid.h"
#include "model/context/Unit.h"
#include "model/entities/EntityFactory.h"
#include "model/entities/CellCluster.h"
#include "model/features/CellFeatureFactory.h"

#include "SimulationAccessImpl.h"

void SimulationAccessImpl::init(SimulationContextApi * context)
{
	_context = static_cast<SimulationContext*>(context);
}

void SimulationAccessImpl::addCell(CellDescription desc)
{
	EntityFactory* entityFactory = ServiceLocator::getInstance().getService<EntityFactory>();
	CellFeatureFactory* featureFactory = ServiceLocator::getInstance().getService<CellFeatureFactory>();
	auto unitContext = _context->getUnitGrid()->getUnitOfMapPos(desc.pos)->getContext();
	
	auto cell = entityFactory->buildCell(desc.energy, unitContext, desc.maxConnections, desc.tokenAccessNumber, QVector3D());
	QList<Cell*> cells;
	cells.push_back(cell);
	featureFactory->addCellFunction(cell, desc.cellFunction.type, desc.cellFunction.data, unitContext);
	featureFactory->addEnergyGuidance(cell, unitContext);

	auto cluster = entityFactory->buildCellCluster(cells, 0.0, desc.pos, 0.0, desc.vel, unitContext);

	_context->getUnitThreadController()->lock();
	unitContext->getClustersRef().push_back(cluster);
	cluster->drawCellsToMap();
	_context->getUnitThreadController()->unlock();
}

