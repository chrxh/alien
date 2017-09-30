#include <QTimer>

#include "Base/ServiceLocator.h"
#include "Model/Context/SimulationContextLocal.h"
#include "Model/Context/UnitThreadController.h"

#include "SimulationControllerImpl.h"

namespace
{
	const double displayFps = 25.0;
}

SimulationControllerImpl::SimulationControllerImpl(QObject* parent)
	: SimulationController(parent)
	, _oneSecondTimer(new QTimer(this))
{
	connect(_oneSecondTimer, &QTimer::timeout, this, &SimulationControllerImpl::oneSecondTimerTimeout);
	_oneSecondTimer->start(1000);
}

void SimulationControllerImpl::init(SimulationContext* context)
{
	SET_CHILD(_context, static_cast<SimulationContextLocal*>(context));
	connect(_context->getUnitThreadController(), &UnitThreadController::timestepCalculated, [this]() {
		Q_EMIT nextTimestepCalculated();
		++_timestepsPerSecond;
		if (_flagSimulationRunning) {
			if (_timeSinceLastStart.elapsed() > (1000.0 / displayFps)*_displayedFramesSinceLastStart) {
				++_displayedFramesSinceLastStart;
				Q_EMIT nextFrameCalculated();
			}
			_context->getUnitThreadController()->calculateTimestep();
		}
		else {
			Q_EMIT nextFrameCalculated();
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

SimulationContext * SimulationControllerImpl::getContext() const
{
	return _context;
}

Q_SLOT void SimulationControllerImpl::oneSecondTimerTimeout()
{
	Q_EMIT updateTimestepsPerSecond(_timestepsPerSecond);
	_timestepsPerSecond = 0;
}




