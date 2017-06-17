#include "GpuWorker.h"
#include "GpuObserver.h"

#include "GpuThreadController.h"

GpuThreadController::GpuThreadController(QObject* parent /*= nullptr*/)
	: QObject(parent)
{
	_worker = new GpuWorker;
	_worker->moveToThread(&_thread);
	connect(this, &GpuThreadController::runSimulationWithGpu, _worker, &GpuWorker::runSimulation);
	connect(_worker, &GpuWorker::timestepCalculated, this, &GpuThreadController::timestepCalculatedWithGpu);
	_thread.start();
}

GpuThreadController::~GpuThreadController()
{
	_worker->setMode(RunningMode::StopAfterNextTimestep);
	_thread.quit();
	if (!_thread.wait(2000)) {
		_thread.terminate();
		_thread.wait();
	}
	delete _worker;
}

void GpuThreadController::init(SpaceMetricApi *metric)
{
	_worker->init(metric);
}

GpuWorker * GpuThreadController::getGpuWorker() const
{
	return _worker;
}

void GpuThreadController::runSimulation(bool run)
{
	if (run) {
		_worker->setMode(RunningMode::OpenEnd);
		if (!_worker->isSimulationRunning()) {
			Q_EMIT runSimulationWithGpu();
		}
	}
	else {
		_worker->setMode(RunningMode::StopAfterNextTimestep);
	}
}

void GpuThreadController::timestepCalculatedWithGpu()
{
	Q_EMIT timestepCalculated();
}

