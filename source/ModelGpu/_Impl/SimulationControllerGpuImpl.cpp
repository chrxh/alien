#include "SimulationContextGpuImpl.h"
#include "SimulationControllerGpuImpl.h"

extern void calcNextTimestepGpu();

void SimulationControllerGpuImpl::init(SimulationContextApi * context)
{
	SET_CHILD(_context, static_cast<SimulationContextGpuImpl*>(context));
}

void SimulationControllerGpuImpl::setRun(bool run)
{
	calcNextTimestepGpu();
}

void SimulationControllerGpuImpl::calculateSingleTimestep()
{
}

SimulationContextApi * SimulationControllerGpuImpl::getContext() const
{
	return _context;
}
