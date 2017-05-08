#include "global/ServiceLocator.h"
#include "model/entities/Descriptions.h"
#include "model/entities/LightDescriptions.h"
#include "model/entities/EntityFactory.h"
#include "model/context/SimulationContext.h"
#include "model/context/UnitContext.h"
#include "model/context/UnitThreadController.h"
#include "model/context/UnitGrid.h"
#include "model/context/Unit.h"
#include "model/entities/CellCluster.h"

#include "SimulationAccessImpl.h"

template class SimulationAccessImpl<DataDescription>;
template class SimulationAccessImpl<DataLightDescription>;

template<typename DataDescriptionType>
SimulationAccessImpl<DataDescriptionType>::~SimulationAccessImpl()
{
	if (_registered) {
		_context->getUnitThreadController()->unregisterObserver(this);
	}
}

template<typename DataDescriptionType>
void SimulationAccessImpl<DataDescriptionType>::init(SimulationContextApi * context)
{
	_context = static_cast<SimulationContext*>(context);
	_context->getUnitThreadController()->registerObserver(this);
	_registered = true;
}

template<typename DataDescriptionType>
void SimulationAccessImpl<DataDescriptionType>::addData(DataDescriptionType const& desc)
{
	_dataToAdd.clusters.insert(_dataToAdd.clusters.end(), desc.clusters.begin(), desc.clusters.end());
	_dataToAdd.particles.insert(_dataToAdd.particles.end(), desc.particles.begin(), desc.particles.end());
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
void SimulationAccessImpl<DataDescriptionType>::requireData(IntRect rect)
{
	_dataRequired = true;
	_requiredRect = rect;
}

template<typename DataDescriptionType>
DataDescriptionType const& SimulationAccessImpl<DataDescriptionType>::retrieveData()
{
	_dataRequired = false;
	return _dataCollected;
}

template<typename DataDescriptionType>
void SimulationAccessImpl<DataDescriptionType>::unregister()
{
	_registered = false;
}

template<typename DataDescriptionType>
void SimulationAccessImpl<DataDescriptionType>::accessToUnits()
{
	callBackAddData();
	callBackGetData();
}

template<typename DataDescriptionType>
void SimulationAccessImpl<DataDescriptionType>::callBackAddData()
{
	EntityFactory* factory = ServiceLocator::getInstance().getService<EntityFactory>();

	auto grid = _context->getUnitGrid();
	for (auto const& clusterDesc : _dataToAdd.clusters) {
		auto unitContext = grid->getUnitOfMapPos(clusterDesc.pos)->getContext();
		auto cluster = factory->build(clusterDesc, unitContext);
		unitContext->getClustersRef().push_back(cluster);
	}
	for (auto const& particleDesc : _dataToAdd.particles) {
		auto unitContext = grid->getUnitOfMapPos(particleDesc.pos)->getContext();
		auto particle = factory->build(particleDesc, unitContext);
		unitContext->getEnergyParticlesRef().push_back(particle);
	}
	_dataToAdd = DataDescriptionType();
}

template<typename DataDescriptionType>
void SimulationAccessImpl<DataDescriptionType>::callBackGetData()
{
	if (!_dataRequired) {
		return;
	}

	auto grid = _context->getUnitGrid();
	IntVector2D gridPosUpperLeft = grid->getGridPosOfMapPos(_requiredRect.p1.toQVector2D());
	IntVector2D gridPosLowerRight = grid->getGridPosOfMapPos(_requiredRect.p2.toQVector2D());
}
