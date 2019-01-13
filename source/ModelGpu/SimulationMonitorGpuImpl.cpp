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
	_data.numParticles= access.numParticles;
}
