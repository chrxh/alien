#include <functional>
#include <QImage>
#include <QElapsedTimer>
#include <QThread>

#include "ModelBasic/SpaceProperties.h"

#include "CudaInterface.cuh"
#include "CudaJob.h"
#include "CudaWorker.h"

CudaWorker::~CudaWorker()
{
	delete _cudaSimulation;
}

void CudaWorker::init(SpaceProperties* space)
{
	_space = space;
	auto size = space->getSize();
	delete _cudaSimulation;
	_cudaSimulation = new CudaSimulation({ size.x, size.y });
}

void CudaWorker::terminateWorker()
{
	std::lock_guard<std::mutex> lock(_mutex);
	_terminate = true;
	_condition.notify_all();
}

void CudaWorker::addJob(CudaJob const & job)
{
	std::lock_guard<std::mutex> lock(_mutex);
	_jobs.push_back(job);
	_condition.notify_all();
}

vector<CudaJob> CudaWorker::getFinishedJobs(string const & originId)
{
	std::lock_guard<std::mutex> lock(_mutex);
	vector<CudaJob> result;
	vector<CudaJob> remainingJobs;
	for (auto const& job : _finishedJobs) {
		if (job->getOriginId() == originId) {
			result.push_back(job);
		}
		else {
			remainingJobs.push_back(job);
		}
	}
	_finishedJobs = remainingJobs;
	return result;
}

void CudaWorker::run()
{
	do {
		QElapsedTimer timer;
		timer.start();

		processJobs();

		if (isSimulationRunning()) {
			_cudaSimulation->calcNextTimestep();
			if (_tpsRestriction) {
				int remainingTime = 1000000 / (*_tpsRestriction) - timer.nsecsElapsed() / 1000;
				if (remainingTime > 0) {
					QThread::usleep(remainingTime);
				}
			}
			Q_EMIT timestepCalculated();
		}

		std::unique_lock<std::mutex> uniqueLock(_mutex);
		if (!_jobs.empty() && !_terminate) {
			_condition.wait(uniqueLock, [this]() {
				return !_jobs.empty() || _terminate;
			});
		}
	} while (!isTerminate());
}

void CudaWorker::processJobs()
{
	std::lock_guard<std::mutex> lock(_mutex);
	if (_jobs.empty()) {
		return;
	}
	bool notify = false;

	for (auto const& job : _jobs) {

		if (auto getDataJob = boost::dynamic_pointer_cast<_GetDataJob>(job)) {
			auto rect = getDataJob->getRect();
			auto dataTO = getDataJob->getDataTO();
			_cudaSimulation->getSimulationData({ rect.p1.x, rect.p1.y }, { rect.p2.x, rect.p2.y }, dataTO);
		}

		if (auto setDataJob = boost::dynamic_pointer_cast<_SetDataJob>(job)) {
			auto rect = setDataJob->getRect();
			auto dataTO = setDataJob->getDataTO();
			_cudaSimulation->setSimulationData({ rect.p1.x, rect.p1.y }, { rect.p2.x, rect.p2.y }, dataTO);
		}

		if (auto runSimJob = boost::dynamic_pointer_cast<_RunSimulationJob>(job)) {
			_simulationRunning = true;
		}

		if (auto stopSimulationJob = boost::dynamic_pointer_cast<_StopSimulationJob>(job)) {
			_simulationRunning = false;
		}

		if (auto calcSingleTimestepJob = boost::dynamic_pointer_cast<_CalcSingleTimestepJob>(job)) {
			_cudaSimulation->calcNextTimestep();
			Q_EMIT timestepCalculated();
		}

		if (auto tpsRestrictionJob = boost::dynamic_pointer_cast<_TpsRestrictionJob>(job)) {
			_tpsRestriction = tpsRestrictionJob->getTpsRestriction();
		}

		if (job->isNotifyFinish()) {
			notify = true;
		}
	}
	if (notify) {
		_finishedJobs.insert(_finishedJobs.end(), _jobs.begin(), _jobs.end());
		_jobs.clear();
		Q_EMIT jobsFinished();
	}
	else {
		_jobs.clear();
	}
}

bool CudaWorker::isTerminate()
{
	std::lock_guard<std::mutex> lock(_mutex);
	return _terminate;
}

bool CudaWorker::isSimulationRunning()
{
	std::lock_guard<std::mutex> lock(_mutex);
	return _simulationRunning;
}
