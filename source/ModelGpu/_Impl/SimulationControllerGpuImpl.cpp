#include <QTimer>

#include "ThreadController.h"
#include "SimulationContextGpuImpl.h"
#include "SimulationControllerGpuImpl.h"


namespace
{
	const int updateFrameInMilliSec = 50.0;
}

SimulationControllerGpuImpl::SimulationControllerGpuImpl(QObject* parent /*= nullptr*/)
	: SimulationController(parent)
	, _oneSecondTimer(new QTimer(this))
	, _frameTimer(new QTimer(this))
{
	connect(_oneSecondTimer, &QTimer::timeout, this, &SimulationControllerGpuImpl::oneSecondTimerTimeout);
	connect(_frameTimer, &QTimer::timeout, this, &SimulationControllerGpuImpl::frameTimerTimeout);

	_oneSecondTimer->start(1000);
	_frameTimer->start(updateFrameInMilliSec);
}

void SimulationControllerGpuImpl::init(SimulationContext * context)
{
	SET_CHILD(_context, static_cast<SimulationContextGpuImpl*>(context));
	connect(_context->getGpuThreadController(), &ThreadController::timestepCalculated, [this]() {
		Q_EMIT nextTimestepCalculated();
		++_timestepsPerSecond;
		if (_mode == RunningMode::OpenEndedSimulation) {
			if (_timeSinceLastStart.elapsed() > updateFrameInMilliSec*_displayedFramesSinceLastStart) {
				++_displayedFramesSinceLastStart;
			}
		}

		if (_mode != RunningMode::OpenEndedSimulation) {
			Q_EMIT nextFrameCalculated();
			_mode = RunningMode::DoNothing;
		}

	});
}

void SimulationControllerGpuImpl::setRun(bool run)
{
	_displayedFramesSinceLastStart = 0;
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

SimulationContext * SimulationControllerGpuImpl::getContext() const
{
	return _context;
}

uint SimulationControllerGpuImpl::getTimestep() const
{
	return 0;
}

void SimulationControllerGpuImpl::oneSecondTimerTimeout()
{
	Q_EMIT updateTimestepsPerSecond(_timestepsPerSecond);
	_timestepsPerSecond = 0;
}

void SimulationControllerGpuImpl::frameTimerTimeout()
{
	if (_mode != RunningMode::DoNothing) {
		Q_EMIT nextFrameCalculated();
	}
}
