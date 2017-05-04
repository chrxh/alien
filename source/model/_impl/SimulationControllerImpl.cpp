#include "global/ServiceLocator.h"
#include "model/context/SimulationContext.h"
#include "model/context/UnitThreadController.h"
#include "model/tools/ToolFactory.h"
#include "model/tools/MapManipulator.h"

#include "SimulationControllerImpl.h"

SimulationControllerImpl::SimulationControllerImpl(QObject* parent)
	: SimulationController(parent)
{
}

void SimulationControllerImpl::init(SimulationContextHandle* context)
{
	SET_CHILD(_context, static_cast<SimulationContext*>(context));
	ToolFactory* factory = ServiceLocator::getInstance().getService<ToolFactory>();
	auto manipulator = factory->buildMapManipulator();
	SET_CHILD(_manipulator, manipulator);
}

void SimulationControllerImpl::setRun(bool run)
{
	if (run) {
		_context->getUnitThreadController()->start();
	}
}

MapManipulator * SimulationControllerImpl::getMapManipulator() const
{
	return nullptr;
}



