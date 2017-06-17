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
	, _oneSecondTimer(new QTimer(this))
{
	connect(_oneSecondTimer, &QTimer::timeout, this, &SimulationControllerGpuImpl::oneSecondTimerTimeout);
	_oneSecondTimer->start(1000);
}

void SimulationControllerGpuImpl::init(SimulationContextApi * context)
{
	SET_CHILD(_context, static_cast<SimulationContextGpuImpl*>(context));
	connect(_context->getGpuThreadController(), &GpuThreadController::timestepCalculated, [this]() {
		Q_EMIT nextTimestepCalculated();
		++_timestepsPerSecond;
		if (_flagSimulationRunning) {
			if (_timeSinceLastStart.elapsed() > (1000.0 / displayFps)*_displayedFramesSinceLastStart) {
				++_displayedFramesSinceLastStart;
				Q_EMIT nextFrameCalculated();
			}
		}
		else {
			Q_EMIT nextFrameCalculated();
		}
	});

}


void SimulationControllerGpuImpl::setRun(bool run)
{
	_displayedFramesSinceLastStart = 0;
	_flagSimulationRunning = run;
	if (run) {
		_timeSinceLastStart.restart();
		_context->getGpuThreadController()->runSimulation(true);
	}
}

void SimulationControllerGpuImpl::calculateSingleTimestep()
{
	_timeSinceLastStart.restart();
	_context->getGpuThreadController()->runSimulation(false);
}

SimulationContextApi * SimulationControllerGpuImpl::getContext() const
{
	return _context;
}

Q_SLOT void SimulationControllerGpuImpl::oneSecondTimerTimeout()
{
	Q_EMIT updateTimestepsPerSecond(_timestepsPerSecond);
	_timestepsPerSecond = 0;
}
