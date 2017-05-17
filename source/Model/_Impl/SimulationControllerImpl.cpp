#include <QTimer>

#include "Base/ServiceLocator.h"
#include "model/Context/SimulationContext.h"
#include "model/Context/UnitThreadController.h"

#include "SimulationControllerImpl.h"

namespace
{
	const double displayFps = 30.0;
}

SimulationControllerImpl::SimulationControllerImpl(QObject* parent)
	: SimulationController(parent)
	, _oneSecondTimer(new QTimer(this))
{
	connect(_oneSecondTimer, &QTimer::timeout, this, &SimulationControllerImpl::oneSecondTimerTimeout);
	_oneSecondTimer->start(1000);
}

void SimulationControllerImpl::init(SimulationContextApi* context)
{
	SET_CHILD(_context, static_cast<SimulationContext*>(context));
	connect(_context->getUnitThreadController(), &UnitThreadController::timestepCalculated, [this]() {
		++_fps;
		if (_flagSimulationRunning) {
			_context->getUnitThreadController()->calculateTimestep();
			if (_timeSinceLastStart.elapsed() > (1000.0 / displayFps)*_displayedFramesSinceLastStart) {
				++_displayedFramesSinceLastStart;
				Q_EMIT timestepCalculated();
			}
		}
		else {
			Q_EMIT timestepCalculated();
		}
	});

	_context->getUnitThreadController()->start();
}

void SimulationControllerImpl::setRun(bool run)
{
	_displayedFramesSinceLastStart = 0;
	_flagSimulationRunning = run;
	if (run) {
		_timeSinceLastStart.restart();
		_context->getUnitThreadController()->calculateTimestep();
	}
}

void SimulationControllerImpl::calculateSingleTimestep()
{
	_timeSinceLastStart.restart();
	_context->getUnitThreadController()->calculateTimestep();
}

SimulationContextApi * SimulationControllerImpl::getContext() const
{
	return _context;
}

Q_SLOT void SimulationControllerImpl::oneSecondTimerTimeout()
{
	Q_EMIT updateFps(_fps);
	_fps = 0;
}




