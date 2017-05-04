#include "global/ServiceLocator.h"
#include "model/context/SimulationContext.h"
#include "model/context/UnitThreadController.h"
#include "model/tools/ToolFactory.h"
#include "model/tools/SimulationManipulator.h"

#include "SimulationControllerImpl.h"

SimulationControllerImpl::SimulationControllerImpl(QObject* parent)
	: SimulationController(parent)
{
}

void SimulationControllerImpl::init(SimulationContextApi* context)
{
	SET_CHILD(_context, static_cast<SimulationContext*>(context));
}

void SimulationControllerImpl::setRun(bool run)
{
	if (run) {
		_context->getUnitThreadController()->start();
	}
}




