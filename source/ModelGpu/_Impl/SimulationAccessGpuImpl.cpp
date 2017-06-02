#include "GpuWorker.h"
#include "GpuThreadController.h"
#include "SimulationContextGpuImpl.h"
#include "SimulationAccessGpuImpl.h"

SimulationAccessGpuImpl::~SimulationAccessGpuImpl()
{
	if (_registered) {
		_context->getGpuThreadController()->unregisterObserver(this);
	}
}

void SimulationAccessGpuImpl::init(SimulationContextApi * context)
{
	_context = static_cast<SimulationContextGpuImpl*>(context);
	_context->getGpuThreadController()->registerObserver(this);
	_registered = true;
}

void SimulationAccessGpuImpl::updateData(DataDescription const & desc)
{
}

void SimulationAccessGpuImpl::requireData(IntRect rect, ResolveDescription const & resolveDesc)
{
	_dataRequired = true;
	_requiredRect = rect;
	_resolveDesc = resolveDesc;

	if(!_context->getGpuThreadController()->isGpuThreadWorking()) {
		accessToUnits();
	}
}

DataDescription const & SimulationAccessGpuImpl::retrieveData()
{
	return _dataCollected;
}

void SimulationAccessGpuImpl::unregister()
{
	_registered = false;
}

void SimulationAccessGpuImpl::accessToUnits()
{
	if (!_dataRequired) {
		return;
	}

	_dataRequired = false;
	_context->getGpuThreadController()->getGpuWorker()->getData(_requiredRect, _resolveDesc, _dataCollected);

	Q_EMIT dataReadyToRetrieve();
}

