#include <sstream>

#include "ModelBasic/Physics.h"
#include "ModelBasic/SpaceProperties.h"

#include "SimulationContextGpuImpl.h"
#include "SimulationControllerGpu.h"
#include "CudaController.h"
#include "CudaWorker.h"
#include "CudaJob.h"
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

    delete _dataTO.numClusters;
    delete _dataTO.numCells;
    delete _dataTO.numParticles;
    delete _dataTO.numTokens;
    delete[] _dataTO.clusters;
    delete[] _dataTO.cells;
    delete[] _dataTO.particles;
    delete[] _dataTO.tokens;

    auto cudaConstants = ModelGpuData(_context->getSpecificData()).getCudaConstants();
    _dataTO.numClusters = new int;
    _dataTO.numCells = new int;
    _dataTO.numParticles = new int;
    _dataTO.numTokens = new int;
    _dataTO.clusters = new ClusterAccessTO[cudaConstants.MAX_CLUSTERS];
    _dataTO.cells = new CellAccessTO[cudaConstants.MAX_CELLS];
    _dataTO.particles = new ParticleAccessTO[cudaConstants.MAX_PARTICLES];
    _dataTO.tokens = new TokenAccessTO[cudaConstants.MAX_TOKENS];
    
    auto cudaBridge = _context->getCudaController()->getCudaWorker();

    for (auto const& connection : _connections) {
		QObject::disconnect(connection);
	}
	_connections.push_back(connect(cudaBridge, &CudaWorker::jobsFinished, this, &SimulationMonitorGpuImpl::jobsFinished, Qt::QueuedConnection));
}

void SimulationMonitorGpuImpl::requireData()
{
	auto cudaWorker = _context->getCudaController()->getCudaWorker();
	auto size = _context->getSpaceProperties()->getSize();
	CudaJob job = boost::make_shared<_GetDataJob>(getObjectId(), IntRect{ { 0, 0 }, size }, _dataTO);
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
		if (auto const& getDataForEditJob = boost::dynamic_pointer_cast<_GetDataJob>(job)) {
			auto dataTO = getDataForEditJob->getDataTO();
			calcMonitorData(dataTO);
			Q_EMIT dataReadyToRetrieve();
		}
	}
}

void SimulationMonitorGpuImpl::calcMonitorData(DataAccessTO const& dataTO)
{
	_monitorData.numCells = *dataTO.numCells;
	_monitorData.numClusters = *dataTO.numClusters;
	_monitorData.numParticles = *dataTO.numParticles;
	_monitorData.numTokens = *dataTO.numTokens;

	_monitorData.totalInternalEnergy = 0.0;
	_monitorData.totalLinearKineticEnergy = 0.0;
	_monitorData.totalRotationalKineticEnergy = 0.0;
	for (int i = 0; i < *dataTO.numClusters; ++i) {
		auto const& cluster = dataTO.clusters[i];
        _monitorData.totalLinearKineticEnergy += Physics::linearKineticEnergy(cluster.numCells, { cluster.vel.x, cluster.vel.y });
//        _monitorData.totalRotationalKineticEnergy += Physics::rotationalKineticEnergy(cluster.angularMass, cluster.angularVel);
	}
	for (int i = 0; i < *dataTO.numCells; ++i) {
		auto const& cell = dataTO.cells[i];
		_monitorData.totalInternalEnergy += cell.energy;
	}
    for (int i = 0; i < *dataTO.numTokens; ++i) {
        auto const& token = dataTO.tokens[i];
        _monitorData.totalInternalEnergy += token.energy;
    }
    for (int i = 0; i < *dataTO.numParticles; ++i) {
        auto const& particle = dataTO.particles[i];
		_monitorData.totalInternalEnergy += particle.energy;
	}
}

string SimulationMonitorGpuImpl::getObjectId() const
{
	auto id = reinterpret_cast<long long>(this);
	std::stringstream stream;
	stream << MonitorGpuId << id;
	return stream.str();
}
