#include "model/context/SimulationContext.h"
#include "model/context/UnitThreadController.h"
#include "SimulationControllerImpl.h"

SimulationControllerImpl::SimulationControllerImpl(QObject* parent)
	: SimulationController(parent)
{
}

void SimulationControllerImpl::init(SimulationContext* context)
{
	if (_context != context) {
		delete _context;
		_context = context;
		_context->setParent(this);
	}
}

void SimulationControllerImpl::setRun(bool run)
{
	if (run) {
		_context->getUnitThreadController()->start();
	}
}



