#include <QTimer>

#include "Base/ServiceLocator.h"
#include "model/Context/SimulationContext.h"
#include "model/Context/UnitThreadController.h"

#include "SimulationControllerImpl.h"

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
	connect(_context->getUnitThreadController(), &UnitThreadController::timestepCalculated, this, &SimulationController::timestepCalculated);
	connect(_context->getUnitThreadController(), &UnitThreadController::timestepCalculated, [this]() {
		if (_flagSimulationRunning) {
			_context->getUnitThreadController()->calculateTimestep();
		}
		++_fps;
	});

	_context->getUnitThreadController()->start();
}

void SimulationControllerImpl::setRun(bool run)
{
	_flagSimulationRunning = run;
	if (run) {
		_context->getUnitThreadController()->calculateTimestep();
	}
}

void SimulationControllerImpl::calculateSingleTimestep()
{
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




