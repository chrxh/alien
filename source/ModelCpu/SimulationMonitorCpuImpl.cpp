
#include "SimulationContextCpuImpl.h"
#include "SimulationControllerCpu.h"
#include "UnitThreadController.h"
#include "UnitGrid.h"
#include "Unit.h"
#include "UnitContext.h"
#include "Cluster.h"
#include "Cell.h"
#include "Particle.h"
#include "Token.h"
#include "ModelCpuData.h"
#include "SimulationMonitorCpuImpl.h"

SimulationMonitorCpuImpl::SimulationMonitorCpuImpl(QObject * parent)
	: SimulationMonitorCpu(parent)
{
	
}

SimulationMonitorCpuImpl::~SimulationMonitorCpuImpl()
{
	if (_registered) {
		_context->getUnitThreadController()->unregisterObserver(this);
	}
}

void SimulationMonitorCpuImpl::init(SimulationControllerCpu* controller)
{
	_context = static_cast<SimulationContextCpuImpl*>(controller->getContext());
	_context->getUnitThreadController()->registerObserver(this);
	_registered = true;
}

void SimulationMonitorCpuImpl::requireData()
{
	_dataRequired = true;
	if (_context->getUnitThreadController()->isNoThreadWorking()) {
		accessToUnits();
	}
}

MonitorData const & SimulationMonitorCpuImpl::retrieveData()
{
	return _data;
}

void SimulationMonitorCpuImpl::unregister()
{
	_registered = false;
}

void SimulationMonitorCpuImpl::accessToUnits()
{
	if (!_dataRequired) {
		return;
	}
	_dataRequired = false;

	calcMonitorData();

	Q_EMIT dataReadyToRetrieve();
}

void SimulationMonitorCpuImpl::calcMonitorData()
{
	_data = MonitorData();
	ModelCpuData data(_context->getSpecificData());
	UnitGrid* grid = _context->getUnitGrid();
	IntVector2D gridSize = data.getGridSize();
	IntVector2D gridPos;
	for (gridPos.x = 0; gridPos.x < gridSize.x; ++gridPos.x) {
		for (gridPos.y = 0; gridPos.y < gridSize.y; ++gridPos.y) {
			calcMonitorDataForUnit(grid->getUnitOfGridPos(gridPos));
		}
	}
}

void SimulationMonitorCpuImpl::calcMonitorDataForUnit(Unit * unit)
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
