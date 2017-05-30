#include "GpuWorker.h"
#include "SimulationContextGpuImpl.h"
#include "SimulationControllerGpuImpl.h"


SimulationControllerGpuImpl::SimulationControllerGpuImpl(QObject* parent /*= nullptr*/)
	: SimulationController(parent)
{
	_worker = new GpuWorker;
	_worker->moveToThread(&_thread);
	connect(&_thread, &QThread::finished, _worker, &QObject::deleteLater);
	connect(this, &SimulationControllerGpuImpl::calculateTimestepWithGpu, _worker, &GpuWorker::calculateTimestep);
	connect(_worker, &GpuWorker::timestepCalculated, this, &SimulationControllerGpuImpl::nextTimestepCalculatedWithGpu);
	_thread.start();
}

SimulationControllerGpuImpl::~SimulationControllerGpuImpl()
{
	_thread.quit();
	_thread.wait();
}

void SimulationControllerGpuImpl::init(SimulationContextApi * context)
{
	SET_CHILD(_context, static_cast<SimulationContextGpuImpl*>(context));
	_worker->init();
}


void SimulationControllerGpuImpl::setRun(bool run)
{
	_flagSimulationRunning = run;
	if (run) {
		Q_EMIT calculateTimestepWithGpu();
	}
}

void SimulationControllerGpuImpl::calculateSingleTimestep()
{
	Q_EMIT calculateTimestepWithGpu();
}

SimulationContextApi * SimulationControllerGpuImpl::getContext() const
{
	return _context;
}

void SimulationControllerGpuImpl::nextTimestepCalculatedWithGpu()
{
	Q_EMIT nextTimestepCalculated();
	if (_flagSimulationRunning) {
		Q_EMIT calculateTimestepWithGpu();
	}
}
