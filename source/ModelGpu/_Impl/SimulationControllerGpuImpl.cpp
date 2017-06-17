#include <QTimer>

#include "GpuThreadController.h"
#include "SimulationContextGpuImpl.h"
#include "SimulationControllerGpuImpl.h"


namespace
{
	const double displayFps = 20.0;
}

SimulationControllerGpuImpl::SimulationControllerGpuImpl(QObject* parent /*= nullptr*/)
	: SimulationController(parent)
	, _nextFrameTimer(new QTimer(this))
{
	connect(_nextFrameTimer, &QTimer::timeout, this, &SimulationControllerGpuImpl::nextFrameTimerTimeout);
	_nextFrameTimer->start(displayFps);
}

void SimulationControllerGpuImpl::init(SimulationContextApi * context)
{
	SET_CHILD(_context, static_cast<SimulationContextGpuImpl*>(context));
	connect(_context->getGpuThreadController(), &GpuThreadController::timestepCalculated, [this]() {
		Q_EMIT nextTimestepCalculated();
		int elapsedTime = _timeSinceLastStart.elapsed();
		if(elapsedTime == 0) {
			elapsedTime = 1;
		}
		Q_EMIT updateTimestepsPerSecond(1000 / elapsedTime);
		_timeSinceLastStart.restart();

		if (_mode != RunningMode::OpenEndedSimulation) {
			Q_EMIT nextFrameCalculated();
			_mode = RunningMode::DoNothing;
		}
	});
}

void SimulationControllerGpuImpl::setRun(bool run)
{
	nextFrameTimerTimeout();
	if (run) {
		_mode = RunningMode::OpenEndedSimulation;
		_timeSinceLastStart.restart();
	}
	else {
		_mode = RunningMode::DoNothing;
	}
	_context->getGpuThreadController()->calculate(_mode);
}

void SimulationControllerGpuImpl::calculateSingleTimestep()
{
	_mode = RunningMode::CalcSingleTimestep;
	_timeSinceLastStart.restart();
	_context->getGpuThreadController()->calculate(_mode);
}

SimulationContextApi * SimulationControllerGpuImpl::getContext() const
{
	return _context;
}

void SimulationControllerGpuImpl::nextFrameTimerTimeout()
{
	if (_mode != RunningMode::DoNothing) {
		Q_EMIT nextFrameCalculated();
	}
}
