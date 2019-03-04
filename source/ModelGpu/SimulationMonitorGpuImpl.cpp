#include "ModelBasic/Physics.h"

#include "SimulationContextGpuImpl.h"
#include "SimulationControllerGpu.h"
#include "ThreadController.h"
#include "CudaWorker.h"

#include "SimulationMonitorGpuImpl.h"

void SimulationMonitorGpuImpl::init(SimulationControllerGpu * controller)
{
	_context = static_cast<SimulationContextGpuImpl*>(controller->getContext());
	auto cudaBridge = _context->getGpuThreadController()->getCudaWorker();

	for (auto const& connection : _connections) {
		QObject::disconnect(connection);
	}
	_connections.push_back(connect(cudaBridge, &CudaWorker::dataObtained, this, &SimulationMonitorGpuImpl::dataObtainedFromGpu, Qt::QueuedConnection));
}

void SimulationMonitorGpuImpl::requireData()
{
	auto cudaBridge = _context->getGpuThreadController()->getCudaWorker();
	cudaBridge->requireData();
}

MonitorData const & SimulationMonitorGpuImpl::retrieveData()
{
	return _data;
}

void SimulationMonitorGpuImpl::dataObtainedFromGpu()
{
	auto cudaBridge = _context->getGpuThreadController()->getCudaWorker();

	cudaBridge->lockData();
	SimulationDataForAccess cudaData = cudaBridge->retrieveData();
	calcMonitorData(cudaData);
	cudaBridge->unlockData();

	Q_EMIT dataReadyToRetrieve();
}

void SimulationMonitorGpuImpl::calcMonitorData(SimulationDataForAccess const& access)
{
	_data.numCells = access.numCells;
	_data.numClusters = access.numClusters;
	_data.numParticles = access.numParticles;

	_data.totalInternalEnergy = 0.0;
	_data.totalLinearKineticEnergy = 0.0;
	_data.totalRotationalKineticEnergy = 0.0;
	for (int i = 0; i < access.numClusters; ++i) {
		ClusterData const& cluster = access.clusters[i];
		_data.totalLinearKineticEnergy += Physics::linearKineticEnergy(cluster.numCells, { cluster.vel.x, cluster.vel.y });
		if (cluster.angularMass < 0) {
			int dummy = 0;
			++dummy;
		}
		_data.totalRotationalKineticEnergy += Physics::rotationalKineticEnergy(cluster.angularMass, cluster.angularVel);
	}
	for (int i = 0; i < access.numCells; ++i) {
		CellData const& cell = access.cells[i];
		_data.totalInternalEnergy += cell.energy;
	}
	for (int i = 0; i < access.numParticles; ++i) {
		ParticleData const& particle = access.particles[i];
		_data.totalInternalEnergy += particle.energy;
	}
}
