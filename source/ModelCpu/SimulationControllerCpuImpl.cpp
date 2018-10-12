#include <QTimer>

#include "Base/ServiceLocator.h"
#include "SimulationContextImpl.h"
#include "UnitThreadController.h"

#include "SimulationControllerCpuImpl.h"

namespace
{
	const double displayFps = 25.0;
}

SimulationControllerCpuImpl::SimulationControllerCpuImpl(QObject* parent)
	: SimulationControllerCpu(parent)
{
	_restrictTpsTimer = new QTimer(this);
	_restrictTpsTimer->setTimerType(Qt::PreciseTimer);
	connect(_restrictTpsTimer, &QTimer::timeout, this, &SimulationControllerCpuImpl::restrictTpsTimerTimeout);
}

void SimulationControllerCpuImpl::init(SimulationContext* context, uint timestep)
{
	_timestep = timestep;
	SET_CHILD(_context, static_cast<SimulationContextImpl*>(context));
	connect(_context->getUnitThreadController(), &UnitThreadController::timestepCalculated, this, &SimulationControllerCpuImpl::nextTimestepCalculatedIntern);
	_context->getUnitThreadController()->start();
}

void SimulationControllerCpuImpl::setRun(bool run)
{
	_displayedFramesSinceLastStart = 0;
	_runMode = run;
	if (run) {
		_timeSinceLastStart.restart();
		_calculationRunning = true;
		_context->getUnitThreadController()->calculateTimestep();
	}
}

void SimulationControllerCpuImpl::calculateSingleTimestep()
{
	_timeSinceLastStart.restart();
	_calculationRunning = true;
	_context->getUnitThreadController()->calculateTimestep();
}

SimulationContext * SimulationControllerCpuImpl::getContext() const
{
	return _context;
}

uint SimulationControllerCpuImpl::getTimestep() const
{
	return _timestep;
}

void SimulationControllerCpuImpl::setRestrictTimestepsPreSecond(optional<int> tps)
{
	_restrictTps = tps;
	if (tps) {
		if (*tps > 0) {
			_restrictTpsTimer->start(1000 / *tps);
		}
		else {
			_restrictTpsTimer->start(1000);
		}
	}
	else {
		_restrictTpsTimer->stop();
		if (_runMode) {
			_context->getUnitThreadController()->calculateTimestep();
		}
	}
}

void SimulationControllerCpuImpl::nextTimestepCalculatedIntern()
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

void SimulationControllerCpuImpl::restrictTpsTimerTimeout()
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



