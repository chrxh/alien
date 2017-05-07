#include "global/ServiceLocator.h"
#include "model/context/SimulationContext.h"
#include "model/context/UnitThreadController.h"

#include "SimulationControllerImpl.h"

SimulationControllerImpl::SimulationControllerImpl(QObject* parent)
	: SimulationController(parent)
{
}

void SimulationControllerImpl::init(SimulationContextApi* context)
{
	SET_CHILD(_context, static_cast<SimulationContext*>(context));
	connect(_context->getUnitThreadController(), &UnitThreadController::timestepCalculated, this, &SimulationController::timestepCalculated);
	connect(_context->getUnitThreadController(), &UnitThreadController::timestepCalculated, [this]() {
		if (_flagSimulationRunning) {
			_context->getUnitThreadController()->calculateTimestep();
		}
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




