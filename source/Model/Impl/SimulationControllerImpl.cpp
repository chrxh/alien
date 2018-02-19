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
}

void SimulationControllerImpl::init(SimulationContext* context, uint timestep)
{
	_timestep = timestep;
	SET_CHILD(_context, static_cast<SimulationContextLocal*>(context));
	connect(_context->getUnitThreadController(), &UnitThreadController::timestepCalculated, [this]() {
		Q_EMIT nextTimestepCalculated();
		++_timestep;
		if (_flagRunMode) {
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
	_flagRunMode = run;
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

uint SimulationControllerImpl::getTimestep() const
{
	return _timestep;
}

void SimulationControllerImpl::setRestrictTimestepsPreSecond(optional<int> tps)
{
}



