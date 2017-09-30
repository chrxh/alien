#include "WorkerForGpu.h"

#include "ThreadController.h"

ThreadController::ThreadController(QObject* parent /*= nullptr*/)
	: QObject(parent)
{
	_worker = new WorkerForGpu;
	_worker->moveToThread(&_thread);
	connect(this, &ThreadController::runSimulationWithGpu, _worker, &WorkerForGpu::runSimulation);
	connect(_worker, &WorkerForGpu::timestepCalculated, this, &ThreadController::timestepCalculatedWithGpu);
	_thread.start();
}

ThreadController::~ThreadController()
{
	_worker->setFlagStopAfterNextTimestep(RunningMode::CalcSingleTimestep);
	_thread.quit();
	if (!_thread.wait(2000)) {
		_thread.terminate();
		_thread.wait();
	}
	delete _worker;
}

void ThreadController::init(SpaceMetric *metric)
{
	_worker->init(metric);
}

WorkerForGpu * ThreadController::getGpuWorker() const
{
	return _worker;
}

void ThreadController::calculate(RunningMode mode)
{
	if (mode == RunningMode::CalcSingleTimestep) {
		_worker->setFlagStopAfterNextTimestep(true);
		if (!_worker->isSimulationRunning()) {
			Q_EMIT runSimulationWithGpu();
		}
	}
	if (mode == RunningMode::OpenEndedSimulation) {
		_worker->setFlagStopAfterNextTimestep(false);
		if (!_worker->isSimulationRunning()) {
			Q_EMIT runSimulationWithGpu();
		}
	}
	if (mode == RunningMode::DoNothing) {
		_worker->setFlagStopAfterNextTimestep(true);
	}
}

void ThreadController::timestepCalculatedWithGpu()
{
	Q_EMIT timestepCalculated();
}

