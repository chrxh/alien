#include "CudaWorker.h"

#include "ThreadController.h"

ThreadController::ThreadController(QObject* parent /*= nullptr*/)
	: QObject(parent)
{
	_worker = new CudaWorker;
	_worker->moveToThread(&_thread);
	connect(this, &ThreadController::runSimulationWithGpu, _worker, &CudaWorker::runSimulation);
	connect(_worker, &CudaWorker::timestepCalculated, this, &ThreadController::timestepCalculatedWithGpu);
	_thread.start();
}

ThreadController::~ThreadController()
{
	_worker->stopAfterNextTimestep(RunningMode::CalcSingleTimestep);
	_thread.quit();
	if (!_thread.wait(2000)) {
		_thread.terminate();
		_thread.wait();
	}
	delete _worker;
}

void ThreadController::init(SpaceProperties *metric)
{
	_worker->init(metric);
}

CudaWorker * ThreadController::getCudaWorker() const
{
	return _worker;
}

void ThreadController::calculate(RunningMode mode)
{
	if (mode == RunningMode::CalcSingleTimestep) {
		_worker->stopAfterNextTimestep(true);
		if (!_worker->isSimulationRunning()) {
			Q_EMIT runSimulationWithGpu();
		}
	}
	if (mode == RunningMode::OpenEndedSimulation) {
		_worker->stopAfterNextTimestep(false);
		if (!_worker->isSimulationRunning()) {
			Q_EMIT runSimulationWithGpu();
		}
	}
	if (mode == RunningMode::DoNothing) {
		_worker->stopAfterNextTimestep(true);
	}
}

void ThreadController::restrictTimestepsPerSecond(optional<int> tps)
{
	_worker->restrictTimestepsPerSecond(tps);
}

void ThreadController::timestepCalculatedWithGpu()
{
	Q_EMIT timestepCalculated();
}

