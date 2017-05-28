#include "SimulationContextGpuImpl.h"
#include "SimulationAccessGpuImpl.h"

void SimulationAccessGpuImpl::init(SimulationContextApi * context)
{
	_context = static_cast<SimulationContextGpuImpl*>(context);
}

void SimulationAccessGpuImpl::updateData(DataDescription const & desc)
{
}

void SimulationAccessGpuImpl::requireData(IntRect rect, ResolveDescription const & resolveDesc)
{
}

DataDescription const & SimulationAccessGpuImpl::retrieveData()
{
	return _dataCollected;
}

