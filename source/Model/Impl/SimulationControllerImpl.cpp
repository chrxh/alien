#include <QTimer>

#include "Base/ServiceLocator.h"
#include "Model/Local/SimulationContextLocal.h"
#include "Model/Local/UnitThreadController.h"

#include "SimulationControllerImpl.h"

namespace
{
	const double displayFps = 25.0;
}

SimulationControllerImpl::SimulationControllerImpl(QObject* parent)
	: SimulationController(parent)
{
	_restrictTpsTimer = new QTimer(this);
	connect(_restrictTpsTimer, &QTimer::timeout, this, &SimulationControllerImpl::restrictTpsTimerTimeout);
}

void SimulationControllerImpl::init(SimulationContext* context, uint timestep)
{
	_timestep = timestep;
	SET_CHILD(_context, static_cast<SimulationContextLocal*>(context));
	connect(_context->getUnitThreadController(), &UnitThreadController::timestepCalculated, this, &SimulationControllerImpl::nextTimestepCalculatedIntern);
	_context->getUnitThreadController()->start();
}

void SimulationControllerImpl::setRun(bool run)
{
	_displayedFramesSinceLastStart = 0;
	_runMode = run;
	if (run) {
		_timeSinceLastStart.restart();
		_calculationRunning = true;
		_context->getUnitThreadController()->calculateTimestep();
	}
}

void SimulationControllerImpl::calculateSingleTimestep()
{
	_timeSinceLastStart.restart();
	_calculationRunning = true;
	_context->getUnitThreadController()->calculateTimestep();
}

SimulationContext * SimulationControllerImpl::getContext() const
{
	return _context;
}

uint SimulationControllerImpl::getTimestep() const
{
	return _timestep;
}

void SimulationControllerImpl::setRestrictTimestepsPreSecond(optional<int> tps)
{
	_restrictTps = tps;
	if (tps) {
		_restrictTpsTimer->start(1000 / *tps);
	}
	else {
		_restrictTpsTimer->stop();
		if (_runMode) {
			_context->getUnitThreadController()->calculateTimestep();
		}
	}
}

void SimulationControllerImpl::nextTimestepCalculatedIntern()
{
	Q_EMIT nextTimestepCalculated();
	++_timestep;
	if (_runMode) {
		if (_timeSinceLastStart.elapsed() > (1000.0 / displayFps)*_displayedFramesSinceLastStart) {
			++_displayedFramesSinceLastStart;
			Q_EMIT nextFrameCalculated();
		}
		if (!_restrictTps || _triggerNewTimestep) {
			_triggerNewTimestep = false;
			_context->getUnitThreadController()->calculateTimestep();
		}
		else {
			_calculationRunning = false;
		}
	}
	else {
		_calculationRunning = false;
		Q_EMIT nextFrameCalculated();
	}
}

void SimulationControllerImpl::restrictTpsTimerTimeout()
{
	if (_runMode && _restrictTps) {
		if (!_calculationRunning) {
			_calculationRunning = true;
			_context->getUnitThreadController()->calculateTimestep();
		}
		else {
			_triggerNewTimestep = true;
		}
	}
}



