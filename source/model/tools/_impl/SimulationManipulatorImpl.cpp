#include "SimulationManipulatorImpl.h"
#include "model/context/SimulationContext.h"

void SimulationManipulatorImpl::init(SimulationContextApi * context)
{
	_context = static_cast<SimulationContext*>(context);
}

void SimulationManipulatorImpl::addCell(CellDescription desc)
{
}
