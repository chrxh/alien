#include "global/ServiceLocator.h"
#include "model/context/SimulationContext.h"
#include "model/context/UnitContext.h"
#include "model/context/UnitThreadController.h"
#include "model/context/UnitGrid.h"
#include "model/context/Unit.h"
#include "model/entities/EntityFactory.h"
#include "model/entities/CellCluster.h"
#include "model/features/CellFeatureFactory.h"
#include "model/AccessPorts/Descriptions.h"
#include "model/AccessPorts/LightDescriptions.h"

#include "SimulationAccessImpl.h"

template class SimulationAccessImpl<DataDescription>;
template class SimulationAccessImpl<DataLightDescription>;

template<typename DataDescriptionType>
void SimulationAccessImpl<DataDescriptionType>::init(SimulationContextApi * context)
{
	_context = static_cast<SimulationContext*>(context);
	SimulationAccessSlotWrapper::init(_context);
}

template<typename DataDescriptionType>
void SimulationAccessImpl<DataDescriptionType>::addData(DataDescriptionType const& desc)
{
	_dataToAdd.clusters.insert(_dataToAdd.clusters.end(), desc.clusters.begin(), desc.clusters.end());
	_dataToAdd.particles.insert(_dataToAdd.particles.end(), desc.particles.begin(), desc.particles.end());

	/*
	EntityFactory* entityFactory = ServiceLocator::getInstance().getService<EntityFactory>();
	CellFeatureFactory* featureFactory = ServiceLocator::getInstance().getService<CellFeatureFactory>();
	auto unitContext = _context->getUnitGrid()->getUnitOfMapPos(desc.relPos)->getContext();

	auto cell = entityFactory->buildCell(desc.energy, unitContext, desc.maxConnections, desc.tokenAccessNumber, QVector3D());
	QList<Cell*> cells;
	cells.push_back(cell);
	featureFactory->addCellFunction(cell, desc.cellFunction.type, desc.cellFunction.data, unitContext);
	featureFactory->addEnergyGuidance(cell, unitContext);

	auto cluster = entityFactory->buildCellCluster(cells, 0.0, desc.relPos, 0.0, desc.vel, unitContext);

	_context->getUnitThreadController()->lock();
	unitContext->getClustersRef().push_back(cluster);
	cluster->drawCellsToMap();
	_context->getUnitThreadController()->unlock();
	*/
}

template<typename DataDescriptionType>
void SimulationAccessImpl<DataDescriptionType>::removeData(DataDescriptionType const & desc)
{
}

template<typename DataDescriptionType>
void SimulationAccessImpl<DataDescriptionType>::updateData(DataDescriptionType const & desc)
{
}

template<typename DataDescriptionType>
void SimulationAccessImpl<DataDescriptionType>::requestData(IntRect rect)
{
}

template<typename DataDescriptionType>
DataDescriptionType const & SimulationAccessImpl<DataDescriptionType>::retrieveData()
{
	return _dataToRetrieve;
}

template<typename DataDescriptionType>
void SimulationAccessImpl<DataDescriptionType>::accessToSimulation()
{
}
