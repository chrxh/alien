#include "model/context/SimulationContext.h"
#include "model/context/UnitThreadController.h"
#include "SimulationControllerImpl.h"

SimulationControllerImpl::SimulationControllerImpl(QObject* parent)
	: SimulationController(parent)
{
}

void SimulationControllerImpl::init(SimulationContext* context)
{
	SET_CHILD(_context, context);
}

void SimulationControllerImpl::setRun(bool run)
{
	if (run) {
		_context->getUnitThreadController()->start();
	}
}



