
#include "Model/Local/SimulationContextLocal.h"
#include "Model/Local/UnitThreadController.h"
#include "Model/Local/UnitGrid.h"
#include "Model/Local/Unit.h"
#include "Model/Local/UnitContext.h"
#include "Model/Local/Cluster.h"
#include "Model/Local/Cell.h"
#include "Model/Local/Particle.h"
#include "Model/Local/Token.h"

#include "SimulationMonitorImpl.h"

SimulationMonitorImpl::SimulationMonitorImpl(QObject * parent)
	: SimulationMonitor(parent)
{
	
}

SimulationMonitorImpl::~SimulationMonitorImpl()
{
	if (_registered) {
		_context->getUnitThreadController()->unregisterObserver(this);
	}
}

void SimulationMonitorImpl::init(SimulationContext * context)
{
	_context = static_cast<SimulationContextLocal*>(context);
	_context->getUnitThreadController()->registerObserver(this);
	_registered = true;
}

void SimulationMonitorImpl::requireData()
{
	_dataRequired = true;
	if (_context->getUnitThreadController()->isNoThreadWorking()) {
		accessToUnits();
	}
}

MonitorData const & SimulationMonitorImpl::retrieveData()
{
	return _data;
}

void SimulationMonitorImpl::unregister()
{
	_registered = false;
}

void SimulationMonitorImpl::accessToUnits()
{
	if (!_dataRequired) {
		return;
	}
	_dataRequired = false;

	calcMonitorData();

	Q_EMIT dataReadyToRetrieve();
}

void SimulationMonitorImpl::calcMonitorData()
{
	_data = MonitorData();
	UnitGrid* grid = _context->getUnitGrid();
	IntVector2D gridSize = _context->getGridSize();
	IntVector2D gridPos;
	for (gridPos.x = 0; gridPos.x < gridSize.x; ++gridPos.x) {
		for (gridPos.y = 0; gridPos.y < gridSize.y; ++gridPos.y) {
			calcMonitorDataForUnit(grid->getUnitOfGridPos(gridPos));
		}
	}
}

void SimulationMonitorImpl::calcMonitorDataForUnit(Unit * unit)
{
	auto const& clusters = unit->getContext()->getClustersRef();
	_data.numClusters += clusters.size();
	for (Cluster* const& cluster : clusters) {
		auto const& cells = cluster->getCellsRef();
		_data.numCells += cells.size();
		_data.totalLinearKineticEnergy += cluster->calcLinearKineticEnergy();
		_data.totalRotationalKineticEnergy += cluster->calcRotationalKineticEnergy();
		for (Cell* const& cell : cells) {
			int numToken = cell->getNumToken();
			_data.numTokens += numToken;
			_data.totalInternalEnergy += cell->getEnergy();
			for (int i = 0; i < numToken; ++i) {
				Token* token = cell->getToken(i);
				_data.totalInternalEnergy += token->getEnergy();
			}
		}
	}
	auto const& particles = unit->getContext()->getParticlesRef();
	_data.numParticles += particles.size();
	for (Particle* const& particle : particles) {
		_data.totalInternalEnergy += particle->getEnergy();
	}
}
