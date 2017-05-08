#include "global/ServiceLocator.h"
#include "model/entities/Descriptions.h"
#include "model/entities/EntityFactory.h"
#include "model/entities/CellCluster.h"
#include "model/context/SimulationContext.h"
#include "model/context/UnitContext.h"
#include "model/context/UnitThreadController.h"
#include "model/context/UnitGrid.h"
#include "model/context/Unit.h"
#include "model/context/SpaceMetric.h"
#include "model/entities/CellCluster.h"

#include "SimulationAccessImpl.h"


SimulationAccessImpl::~SimulationAccessImpl()
{
	if (_registered) {
		_context->getUnitThreadController()->unregisterObserver(this);
	}
}

void SimulationAccessImpl::init(SimulationContextApi * context)
{
	_context = static_cast<SimulationContext*>(context);
	_context->getUnitThreadController()->registerObserver(this);
	_registered = true;
}

void SimulationAccessImpl::updateData(DataDescription const & desc)
{
	_dataToUpdate.clusters.insert(_dataToUpdate.clusters.end(), desc.clusters.begin(), desc.clusters.end());
	_dataToUpdate.particles.insert(_dataToUpdate.particles.end(), desc.particles.begin(), desc.particles.end());
}

void SimulationAccessImpl::requireData(IntRect rect)
{
	_dataRequired = true;
	_requiredRect = rect;
	if (_context->getUnitThreadController()->isNoThreadWorking()) {
		accessToUnits();
	}
}

DataDescription const& SimulationAccessImpl::retrieveData()
{
	_dataRequired = false;
	return _dataCollected;
}

void SimulationAccessImpl::unregister()
{
	_registered = false;
}

void SimulationAccessImpl::accessToUnits()
{
	callBackUpdateData();
	callBackGetData();
}

void SimulationAccessImpl::callBackUpdateData()
{
	EntityFactory* factory = ServiceLocator::getInstance().getService<EntityFactory>();

	auto grid = _context->getUnitGrid();

	for (auto const& clusterDesc : _dataToUpdate.clusters) {
		if (clusterDesc.isAdded()) {
			auto const& clusterDescVal = clusterDesc.getValue();
			auto unitContext = grid->getUnitOfMapPos(clusterDescVal.pos.getValue())->getContext();
			auto cluster = factory->build(clusterDescVal, unitContext);
			unitContext->getClustersRef().push_back(cluster);
		}
	}
	for (auto const& particleDesc : _dataToUpdate.particles) {
		if (particleDesc.isAdded()) {
			auto const& particleDescVel = particleDesc.getValue();
			auto unitContext = grid->getUnitOfMapPos(particleDescVel.pos.getValue())->getContext();
			auto particle = factory->build(particleDescVel, unitContext);
			unitContext->getEnergyParticlesRef().push_back(particle);
		}
	}

	_dataToUpdate.clear();
}

void SimulationAccessImpl::callBackGetData()
{
	if (!_dataRequired) {
		return;
	}

	auto grid = _context->getUnitGrid();
	IntVector2D gridPosUpperLeft = grid->getGridPosOfMapPos(_requiredRect.p1.toQVector2D());
	IntVector2D gridPosLowerRight = grid->getGridPosOfMapPos(_requiredRect.p2.toQVector2D());
	IntVector2D gridPos;
	for (gridPos.x = gridPosUpperLeft.x; gridPos.x <= gridPosLowerRight.x; ++gridPos.x) {
		for (gridPos.y = gridPosUpperLeft.y; gridPos.y <= gridPosLowerRight.y; ++gridPos.y) {
			getDataFromUnit(grid->getUnitOfGridPos(gridPos));
		}
	}
}

void SimulationAccessImpl::getDataFromUnit(Unit * unit)
{
	auto const& clusters = unit->getContext()->getClustersRef();
	auto metric = unit->getContext()->getSpaceMetric();
	for (auto const& cluster : clusters) {
		auto pos = metric->correctPositionWithIntPrecision(cluster->getPosition());
		if (_requiredRect.isContained(pos)) {
			_dataCollected.clusters.push_back(cluster->getDescription());
		}
	}
}
