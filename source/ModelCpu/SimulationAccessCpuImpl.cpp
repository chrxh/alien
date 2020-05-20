#include <QImage>

#include "Base/ServiceLocator.h"
#include "ModelBasic/ChangeDescriptions.h"
#include "ModelBasic/Settings.h"
#include "ModelBasic/SpaceProperties.h"

#include "EntityFactory.h"
#include "Cluster.h"
#include "Cell.h"
#include "Particle.h"
#include "SimulationContextCpuImpl.h"
#include "UnitContext.h"
#include "UnitThreadController.h"
#include "UnitGrid.h"
#include "Unit.h"
#include "ParticleMap.h"
#include "CellMap.h"
#include "Cluster.h"
#include "SimulationControllerCpu.h"

#include "SimulationAccessCpuImpl.h"


SimulationAccessCpuImpl::~SimulationAccessCpuImpl()
{
	if (_registered) {
		_context->getUnitThreadController()->unregisterObserver(this);
	}
}

void SimulationAccessCpuImpl::init(SimulationControllerCpu* controller)
{
	_context = static_cast<SimulationContextCpuImpl*>(controller->getContext());
	_context->getUnitThreadController()->registerObserver(this);
	_registered = true;
}

void SimulationAccessCpuImpl::clear()
{
	_toClear = true;
	if (_context->getUnitThreadController()->isNoThreadWorking()) {
		accessToUnits();
	}
}

void SimulationAccessCpuImpl::updateData(DataChangeDescription const & desc)
{
	_dataToUpdate.clusters.insert(_dataToUpdate.clusters.end(), desc.clusters.begin(), desc.clusters.end());
	_dataToUpdate.particles.insert(_dataToUpdate.particles.end(), desc.particles.begin(), desc.particles.end());
	if (_context->getUnitThreadController()->isNoThreadWorking()) {
		accessToUnits();
	}
}

void SimulationAccessCpuImpl::requireData(IntRect rect, ResolveDescription const& resolveDesc)
{
	_dataRequired = true;
	_requiredRect = rect;
	_resolveDesc = resolveDesc;
	if (_context->getUnitThreadController()->isNoThreadWorking()) {
		accessToUnits();
	}
}

void SimulationAccessCpuImpl::requireData(ResolveDescription const & resolveDesc)
{
    auto const space = _context->getSpaceProperties();
    requireData(IntRect{ {0, 0}, space->getSize() }, resolveDesc);
}

void SimulationAccessCpuImpl::requireImage(IntRect rect, QImagePtr const& target)
{
	_imageRequired = true;
	_requiredRect = rect;
	_requiredImage = target;
	if (_context->getUnitThreadController()->isNoThreadWorking()) {
		accessToUnits();
	}
}

DataDescription const& SimulationAccessCpuImpl::retrieveData()
{
	return _dataCollected;
}

void SimulationAccessCpuImpl::unregister()
{
	_registered = false;
}

void SimulationAccessCpuImpl::accessToUnits()
{
	callBackClear();
	callBackUpdateData();
	callBackCollectData();
}

void SimulationAccessCpuImpl::callBackClear()
{
	if (!_toClear) {
		return;
	}
	auto grid = _context->getUnitGrid();
	IntVector2D gridSize = grid->getSize();
	for (int gridX = 0; gridX < gridSize.x; ++gridX) {
		for (int gridY = 0; gridY < gridSize.y; ++gridY) {
			auto unitContext = grid->getUnitOfGridPos({ gridX, gridY })->getContext();
			unitContext->getClustersRef().clear();
			unitContext->getParticlesRef().clear();
			unitContext->getCellMap()->clear();
			unitContext->getParticleMap()->clear();
		}
	}
	_toClear = false;
}

void SimulationAccessCpuImpl::callBackUpdateData()
{
	if (!_dataToUpdate.empty()) {
		updateClusterData();
		updateParticleData();
		Q_EMIT dataUpdated();
	}
}

void SimulationAccessCpuImpl::updateClusterData()
{
	if (_dataToUpdate.clusters.empty()) {
		return;
	}
	EntityFactory* factory = ServiceLocator::getInstance().getService<EntityFactory>();

	auto grid = _context->getUnitGrid();

	unordered_set<uint64_t> clusterIdsToDelete;
	unordered_map<uint64_t, ClusterChangeDescription> clusterToUpdate;
	unordered_set<UnitContext*> units;

	for (auto const& clusterTracker : _dataToUpdate.clusters) {
		auto const& clusterDesc = clusterTracker.getValue();
		if (clusterTracker.isAdded()) {
			ClusterDescription clusterDescToAdd(clusterDesc);
			if (!clusterDescToAdd.pos) {
				clusterDescToAdd.pos = clusterDescToAdd.getClusterPosFromCells();
			}
			auto unitContext = grid->getUnitOfMapPos(*clusterDescToAdd.pos)->getContext();
			auto cluster = factory->build(clusterDescToAdd, unitContext);
			unitContext->getClustersRef().push_back(cluster);
		}
		if (clusterTracker.isDeleted()) {
			auto unitContext = grid->getUnitOfMapPos(*clusterDesc.pos)->getContext();
			units.insert(unitContext);
			clusterIdsToDelete.insert(clusterDesc.id);
		}
		if (clusterTracker.isModified()) {
			auto unitContext = grid->getUnitOfMapPos(clusterDesc.pos.getOldValue())->getContext();
			units.insert(unitContext);
			clusterToUpdate.insert_or_assign(clusterDesc.id, clusterDesc);
		}
	}

	for (auto const& unitContext : units) {
		QMutableListIterator<Cluster*> clusterIt(unitContext->getClustersRef());
		auto cellMap = unitContext->getParticleMap();
		while (clusterIt.hasNext()) {
			Cluster* cluster = clusterIt.next();
			if (clusterToUpdate.find(cluster->getId()) != clusterToUpdate.end()) {
				ClusterChangeDescription const& change = clusterToUpdate.at(cluster->getId());
				cluster->clearCellsFromMap();
				cluster->applyChangeDescription(change);

				auto newUnitContext = grid->getUnitOfMapPos(cluster->getPosition())->getContext();
				if (newUnitContext != unitContext) {
					clusterIt.remove();
					newUnitContext->getClustersRef().push_back(cluster);
					cluster->setContext(newUnitContext);
				}
				cluster->drawCellsToMap();
				clusterToUpdate.erase(cluster->getId());
			}
			if (clusterIdsToDelete.find(cluster->getId()) != clusterIdsToDelete.end()) {
				cluster->clearCellsFromMap();
				delete cluster;
				clusterIt.remove();
			}
		}
	}
	_dataToUpdate.clusters.clear();
}

void SimulationAccessCpuImpl::updateParticleData()
{
	if (_dataToUpdate.particles.empty()) {
		return;
	}

	EntityFactory* factory = ServiceLocator::getInstance().getService<EntityFactory>();

	auto grid = _context->getUnitGrid();

	unordered_set<uint64_t> particleIdsToDelete;
	unordered_map<uint64_t, ParticleChangeDescription> particlesToUpdate;
	unordered_set<UnitContext*> units;

	for (auto const& particleTracker : _dataToUpdate.particles) {
		auto const& particleDesc = particleTracker.getValue();
		if (particleTracker.isAdded()) {
			auto unitContext = grid->getUnitOfMapPos(*particleDesc.pos)->getContext();
			auto particle = factory->build(particleDesc, unitContext);
			unitContext->getParticlesRef().push_back(particle);
		}
		if (particleTracker.isDeleted()) {
			auto unitContext = grid->getUnitOfMapPos(*particleDesc.pos)->getContext();
			units.insert(unitContext);
			particleIdsToDelete.insert(particleDesc.id);
		}
		if (particleTracker.isModified()) {
			auto unitContext = grid->getUnitOfMapPos(particleDesc.pos.getOldValue())->getContext();
			units.insert(unitContext);
			particlesToUpdate.insert_or_assign(particleDesc.id, particleDesc);
		}
	}

	for (auto const& unitContext : units) {
		QMutableListIterator<Particle*> particleIt(unitContext->getParticlesRef());
		while (particleIt.hasNext()) {
			Particle* particle = particleIt.next();
			if (particlesToUpdate.find(particle->getId()) != particlesToUpdate.end()) {
				particle->clearParticleFromMap();
				ParticleChangeDescription const& change = particlesToUpdate.at(particle->getId());
				particle->applyChangeDescription(change);

				auto newUnitContext = grid->getUnitOfMapPos(particle->getPosition())->getContext();
				if (newUnitContext != unitContext) {
					particleIt.remove();
					newUnitContext->getParticlesRef().push_back(particle);
					particle->setContext(newUnitContext);
				}
				particle->drawParticleToMap();
				particlesToUpdate.erase(particle->getId());
			}
			if (particleIdsToDelete.find(particle->getId()) != particleIdsToDelete.end()) {
				particle->clearParticleFromMap();
				particleIt.remove();
				delete particle;
			}
		}
	}
	_dataToUpdate.particles.clear();
}

namespace
{
	void includeNeighborUnits(IntVector2D& gridPosUpperLeft, IntVector2D& gridPosLowerRight, IntVector2D const& gridSize)
	{
		gridPosUpperLeft.x = std::max(gridPosUpperLeft.x - 1, 0);
		gridPosUpperLeft.y = std::max(gridPosUpperLeft.y - 1, 0);
		gridPosLowerRight.x = std::min(gridPosLowerRight.x + 1, gridSize.x - 1);
		gridPosLowerRight.y = std::min(gridPosLowerRight.y + 1, gridSize.y - 1);
	}
}

void SimulationAccessCpuImpl::callBackCollectData()
{
	if (!_dataRequired) {
		return;
	}

	_dataRequired = false;
	_dataCollected.clear();
	auto grid = _context->getUnitGrid();
	IntVector2D gridPosUpperLeft = grid->getGridPosOfMapPos(_requiredRect.p1.toQVector2D(), UnitGrid::CorrectionMode::Truncation);
	IntVector2D gridPosLowerRight = grid->getGridPosOfMapPos(_requiredRect.p2.toQVector2D(), UnitGrid::CorrectionMode::Truncation);
	includeNeighborUnits(gridPosUpperLeft, gridPosLowerRight, grid->getSize());
	IntVector2D gridPos;
	for (gridPos.x = gridPosUpperLeft.x; gridPos.x <= gridPosLowerRight.x; ++gridPos.x) {
		for (gridPos.y = gridPosUpperLeft.y; gridPos.y <= gridPosLowerRight.y; ++gridPos.y) {
			collectDataFromUnit(grid->getUnitOfGridPos(gridPos));
		}
	}

	Q_EMIT dataReadyToRetrieve();
}

void SimulationAccessCpuImpl::collectDataFromUnit(Unit * unit)
{
	collectClustersFromUnit(unit);
	collectParticlesFromUnit(unit);
}

void SimulationAccessCpuImpl::collectClustersFromUnit(Unit * unit)
{
	auto metric = unit->getContext()->getSpaceProperties();
	auto const& clusters = unit->getContext()->getClustersRef();
	for (auto const& cluster : clusters) {
		bool contained = false;
		for (Cell* cell : cluster->getCellsRef()) {
			if (_requiredRect.isContained(cell->calcPosition(true))) {
				contained = true;
				break;
			}
		}
		IntVector2D pos = cluster->getPosition();
		if (contained) {
			_dataCollected.addCluster(cluster->getDescription(_resolveDesc));
		}
	}
}

void SimulationAccessCpuImpl::collectParticlesFromUnit(Unit * unit)
{
	auto metric = unit->getContext()->getSpaceProperties();
	auto const& particles = unit->getContext()->getParticlesRef();
	for (auto const& particle : particles) {
		IntVector2D pos = particle->getPosition();
		if (_requiredRect.isContained(pos)) {
			_dataCollected.addParticle(particle->getDescription(_resolveDesc));
		}
	}
}
