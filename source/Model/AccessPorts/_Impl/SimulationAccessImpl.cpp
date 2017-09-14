#include <QImage>

#include "Base/ServiceLocator.h"
#include "Model/Entities/ChangeDescriptions.h"
#include "Model/Entities/EntityFactory.h"
#include "Model/Entities/Cluster.h"
#include "Model/Entities/Cell.h"
#include "Model/Entities/Particle.h"
#include "Model/Context/SimulationContext.h"
#include "Model/Context/UnitContext.h"
#include "Model/Context/UnitThreadController.h"
#include "Model/Context/UnitGrid.h"
#include "Model/Context/Unit.h"
#include "Model/Context/SpaceMetric.h"
#include "Model/Context/EnergyParticleMap.h"
#include "Model/Entities/Cluster.h"

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
	EntityFactory* factory = ServiceLocator::getInstance().getService<EntityFactory>();

	auto grid = _context->getUnitGrid();

	{
		unordered_set<uint64_t> clusterIdsToDelete;
		unordered_set<UnitContext*> units;

		for (auto const& clusterTracker : _dataToUpdate.clusters) {
			auto const& clusterDesc = clusterTracker.getValue();
			auto unitContext = grid->getUnitOfMapPos(*clusterDesc.pos)->getContext();
			if (clusterTracker.isAdded()) {
				auto cluster = factory->build(clusterDesc, unitContext);
				unitContext->getClustersRef().push_back(cluster);
			}
			if (clusterTracker.isDeleted()) {
				units.insert(unitContext);
				clusterIdsToDelete.insert(clusterDesc.id);
			}
			if (clusterTracker.isModified()) {
				THROW_NOT_IMPLEMENTED();
			}
		}

		for (auto const& unitContext : units) {
			QMutableListIterator<Cluster*> clusterIt(unitContext->getClustersRef());
			while (clusterIt.hasNext()) {
				Cluster* cluster = clusterIt.next();
				if (clusterIdsToDelete.find(cluster->getId()) != clusterIdsToDelete.end()) {
					cluster->clearCellsFromMap();
					delete cluster;
					clusterIt.remove();
				}
			}
		}
	}

	{
		unordered_set<uint64_t> particleIdsToDelete;
		unordered_map<uint64_t, ParticleChangeDescription> particlesToUpdate;
		unordered_set<UnitContext*> units;

		for (auto const& particleTracker : _dataToUpdate.particles) {
			auto const& particleDesc = particleTracker.getValue();
			auto unitContext = grid->getUnitOfMapPos(*particleDesc.pos)->getContext();
			if (particleTracker.isAdded()) {
				auto particle = factory->build(particleDesc, unitContext);
				unitContext->getParticlesRef().push_back(particle);
			}
			if (particleTracker.isDeleted()) {
				units.insert(unitContext);
				particleIdsToDelete.insert(particleDesc.id);
			}
			if (particleTracker.isModified()) {
				units.insert(unitContext);
				particlesToUpdate.insert_or_assign(particleDesc.id, particleDesc);
			}
		}

		for (auto const& unitContext : units) {
			QMutableListIterator<Particle*> particleIt(unitContext->getParticlesRef());
			auto energyMap = unitContext->getEnergyParticleMap();
			while (particleIt.hasNext()) {
				Particle* particle = particleIt.next();
				if (particleIdsToDelete.find(particle->getId()) != particleIdsToDelete.end()) {
					energyMap->removeParticleIfPresent(particle->getPosition(), particle);
					particleIt.remove();
					delete particle;
				}
				if (particlesToUpdate.find(particle->getId()) != particlesToUpdate.end()) {
					ParticleChangeDescription const& change = particlesToUpdate.at(particle->getId());
					energyMap->removeParticleIfPresent(particle->getPosition(), particle);
					particle->applyChangeDescription(change);
					energyMap->setParticle(particle->getPosition(), particle);
				}
			}
		}
	}
	_dataToUpdate.clear();
}

void SimulationAccessImpl::callBackCollectData()
{
	if (!_dataRequired) {
		return;
	}

	_dataRequired = false;
	_dataCollected.clear();
	auto grid = _context->getUnitGrid();
	IntVector2D gridPosUpperLeft = grid->getGridPosOfMapPos(_requiredRect.p1.toQVector2D());
	IntVector2D gridPosLowerRight = grid->getGridPosOfMapPos(_requiredRect.p2.toQVector2D());
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
	IntVector2D gridPosUpperLeft = grid->getGridPosOfMapPos(_requiredRect.p1.toQVector2D());
	IntVector2D gridPosLowerRight = grid->getGridPosOfMapPos(_requiredRect.p2.toQVector2D());
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
				_requiredImage->setPixel(pos.x, pos.y, 0xFF);
			}
		}
	}
}

void SimulationAccessImpl::drawParticlesFromUnit(Unit * unit)
{
	auto metric = unit->getContext()->getSpaceMetric();
	auto const &particles = unit->getContext()->getParticlesRef();
	for (auto const &particle : particles) {
		auto pos = metric->correctPositionAndConvertToIntVector(particle->getPosition());
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
		auto pos = metric->correctPositionAndConvertToIntVector(cluster->getPosition());
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
		auto pos = metric->correctPositionAndConvertToIntVector(particle->getPosition());
		if (_requiredRect.isContained(pos)) {
			_dataCollected.addParticle(particle->getDescription());
		}
	}
}
