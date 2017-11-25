#include <QImage>

#include "Base/ServiceLocator.h"
#include "Model/Api/ChangeDescriptions.h"
#include "Model/Local/EntityFactory.h"
#include "Model/Local/Cluster.h"
#include "Model/Local/Cell.h"
#include "Model/Local/Particle.h"
#include "Model/Local/SimulationContextLocal.h"
#include "Model/Local/UnitContext.h"
#include "Model/Local/UnitThreadController.h"
#include "Model/Local/UnitGrid.h"
#include "Model/Local/Unit.h"
#include "Model/Local/SpaceMetricLocal.h"
#include "Model/Local/ParticleMap.h"
#include "Model/Local/Cluster.h"

#include "SimulationAccessImpl.h"


SimulationAccessImpl::~SimulationAccessImpl()
{
	if (_registered) {
		_context->getUnitThreadController()->unregisterObserver(this);
	}
}

void SimulationAccessImpl::init(SimulationContext * context)
{
	_context = static_cast<SimulationContextLocal*>(context);
	_context->getUnitThreadController()->registerObserver(this);
	_registered = true;
}

void SimulationAccessImpl::updateData(DataChangeDescription const & desc)
{
	_dataToUpdate.clusters.insert(_dataToUpdate.clusters.end(), desc.clusters.begin(), desc.clusters.end());
	_dataToUpdate.particles.insert(_dataToUpdate.particles.end(), desc.particles.begin(), desc.particles.end());
	if (_context->getUnitThreadController()->isNoThreadWorking()) {
		accessToUnits();
	}
}

void SimulationAccessImpl::requireData(IntRect rect, ResolveDescription const& resolveDesc)
{
	_dataRequired = true;
	_requiredRect = rect;
	_resolveDesc = resolveDesc;
	if (_context->getUnitThreadController()->isNoThreadWorking()) {
		accessToUnits();
	}
}

void SimulationAccessImpl::requireImage(IntRect rect, QImage * target)
{
	_imageRequired = true;
	_requiredRect = rect;
	_requiredImage = target;
	if (_context->getUnitThreadController()->isNoThreadWorking()) {
		accessToUnits();
	}
}

DataDescription const& SimulationAccessImpl::retrieveData()
{
	return _dataCollected;
}

void SimulationAccessImpl::unregister()
{
	_registered = false;
}

void SimulationAccessImpl::accessToUnits()
{
	callBackUpdateData();
	callBackCollectData();
	callBackDrawImage();
}

void SimulationAccessImpl::callBackUpdateData()
{
	updateClusterData();
	updateParticleData();

	_dataToUpdate.clear();
}

void SimulationAccessImpl::updateClusterData()
{
	EntityFactory* factory = ServiceLocator::getInstance().getService<EntityFactory>();

	auto grid = _context->getUnitGrid();

	unordered_set<uint64_t> clusterIdsToDelete;
	unordered_map<uint64_t, ClusterChangeDescription> clusterToUpdate;
	unordered_set<UnitContext*> units;

	for (auto const& clusterTracker : _dataToUpdate.clusters) {
		auto const& clusterDesc = clusterTracker.getValue();
		if (clusterTracker.isAdded()) {
			auto unitContext = grid->getUnitOfMapPos(*clusterDesc.pos)->getContext();
			auto cluster = factory->build(clusterDesc, unitContext);
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
}

void SimulationAccessImpl::updateParticleData()
{
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
}

void SimulationAccessImpl::callBackCollectData()
{
	if (!_dataRequired) {
		return;
	}

	_dataRequired = false;
	_dataCollected.clear();
	auto grid = _context->getUnitGrid();
	IntVector2D gridPosUpperLeft = grid->getGridPosOfMapPos(_requiredRect.p1.toQVector2D(), UnitGrid::CorrectionMode::Truncation);
	IntVector2D gridPosLowerRight = grid->getGridPosOfMapPos(_requiredRect.p2.toQVector2D(), UnitGrid::CorrectionMode::Truncation);
	IntVector2D gridPos;
	for (gridPos.x = gridPosUpperLeft.x; gridPos.x <= gridPosLowerRight.x; ++gridPos.x) {
		for (gridPos.y = gridPosUpperLeft.y; gridPos.y <= gridPosLowerRight.y; ++gridPos.y) {
			collectDataFromUnit(grid->getUnitOfGridPos(gridPos));
		}
	}

	Q_EMIT dataReadyToRetrieve();
}

void SimulationAccessImpl::callBackDrawImage()
{
	if (!_imageRequired) {
		return;
	}
	_imageRequired = false;

	_requiredImage->fill(QColor(0x00, 0x00, 0x1b));

	auto grid = _context->getUnitGrid();
	IntVector2D gridPosUpperLeft = grid->getGridPosOfMapPos(_requiredRect.p1.toQVector2D(), UnitGrid::CorrectionMode::Truncation);
	IntVector2D gridPosLowerRight = grid->getGridPosOfMapPos(_requiredRect.p2.toQVector2D(), UnitGrid::CorrectionMode::Truncation);
	IntVector2D gridPos;
	for (gridPos.x = gridPosUpperLeft.x; gridPos.x <= gridPosLowerRight.x; ++gridPos.x) {
		for (gridPos.y = gridPosUpperLeft.y; gridPos.y <= gridPosLowerRight.y; ++gridPos.y) {
			drawImageFromUnit(grid->getUnitOfGridPos(gridPos));
		}
	}

	Q_EMIT imageReady();
}

void SimulationAccessImpl::drawImageFromUnit(Unit * unit)
{
	drawClustersFromUnit(unit);
	drawParticlesFromUnit(unit);
}

void SimulationAccessImpl::drawClustersFromUnit(Unit * unit)
{
	auto metric = unit->getContext()->getSpaceMetric();
	auto const &clusters = unit->getContext()->getClustersRef();
	for (auto const &cluster : clusters) {
		for (auto const &cell : cluster->getCellsRef()) {
			auto pos = metric->correctPositionAndConvertToIntVector(cell->calcPosition(true));
			if (_requiredRect.isContained(pos)) {
				if (cell->getNumToken() > 0) {
					_requiredImage->setPixel(pos.x, pos.y, 0xFFFFFF);
				} else {
					_requiredImage->setPixel(pos.x, pos.y, 0xFF);
				}
			}
		}
	}
}

void SimulationAccessImpl::drawParticlesFromUnit(Unit * unit)
{
	auto metric = unit->getContext()->getSpaceMetric();
	auto const &particles = unit->getContext()->getParticlesRef();
	for (auto const &particle : particles) {
		IntVector2D pos = particle->getPosition();
		if (_requiredRect.isContained(pos)) {
			_requiredImage->setPixel(pos.x, pos.y, 0x902020);
		}
	}
}

void SimulationAccessImpl::collectDataFromUnit(Unit * unit)
{
	collectClustersFromUnit(unit);
	collectParticlesFromUnit(unit);
}

void SimulationAccessImpl::collectClustersFromUnit(Unit * unit)
{
	auto metric = unit->getContext()->getSpaceMetric();
	auto const& clusters = unit->getContext()->getClustersRef();
	for (auto const& cluster : clusters) {
		IntVector2D pos = cluster->getPosition();
		if (_requiredRect.isContained(pos)) {
			_dataCollected.addCluster(cluster->getDescription(_resolveDesc));
		}
	}
}

void SimulationAccessImpl::collectParticlesFromUnit(Unit * unit)
{
	auto metric = unit->getContext()->getSpaceMetric();
	auto const& particles = unit->getContext()->getParticlesRef();
	for (auto const& particle : particles) {
		IntVector2D pos = particle->getPosition();
		if (_requiredRect.isContained(pos)) {
			_dataCollected.addParticle(particle->getDescription());
		}
	}
}
