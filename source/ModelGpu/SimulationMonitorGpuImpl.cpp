#include <sstream>

#include "ModelBasic/Physics.h"
#include "ModelBasic/SpaceProperties.h"

#include "SimulationContextGpuImpl.h"
#include "SimulationControllerGpu.h"
#include "CudaController.h"
#include "CudaWorker.h"
#include "CudaJobs.h"
#include "CudaConstants.h"
#include "SimulationMonitorGpuImpl.h"

namespace
{
	const string MonitorGpuId = "MonitorGpuId";
}

SimulationMonitorGpuImpl::SimulationMonitorGpuImpl(QObject* parent /*= nullptr*/)
	: SimulationMonitorGpu(parent)
{
}

SimulationMonitorGpuImpl::~SimulationMonitorGpuImpl()
{
}

void SimulationMonitorGpuImpl::init(SimulationControllerGpu * controller)
{
    _context = static_cast<SimulationContextGpuImpl*>(controller->getContext());

    auto cudaWorker = _context->getCudaController()->getCudaWorker();

    for (auto const& connection : _connections) {
		QObject::disconnect(connection);
	}
	_connections.push_back(connect(cudaWorker, &CudaWorker::jobsFinished, this, &SimulationMonitorGpuImpl::jobsFinished, Qt::QueuedConnection));
}

void SimulationMonitorGpuImpl::requireData()
{
	auto const cudaWorker = _context->getCudaController()->getCudaWorker();
    auto const job = boost::make_shared<_GetMonitorDataJob>(getObjectId());
    cudaWorker->addJob(job);
}

MonitorData const & SimulationMonitorGpuImpl::retrieveData()
{
	return _monitorData;
}

void SimulationMonitorGpuImpl::jobsFinished()
{
	auto worker = _context->getCudaController()->getCudaWorker();
	auto finishedJobs = worker->getFinishedJobs(getObjectId());
	for (auto const& job : finishedJobs) {
		if (auto const& getMonitorDataJob = boost::dynamic_pointer_cast<_GetMonitorDataJob>(job)) {
            _monitorData = getMonitorDataJob->getMonitorData();
			Q_EMIT dataReadyToRetrieve();
		}
	}
}

string SimulationMonitorGpuImpl::getObjectId() const
{
	auto id = reinterpret_cast<long long>(this);
	std::stringstream stream;
	stream << MonitorGpuId << id;
	return stream.str();
}
