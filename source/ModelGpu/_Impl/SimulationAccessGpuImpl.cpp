#include "GpuWorker.h"
#include "GpuThreadController.h"
#include "SimulationContextGpuImpl.h"
#include "SimulationAccessGpuImpl.h"

SimulationAccessGpuImpl::~SimulationAccessGpuImpl()
{
}

void SimulationAccessGpuImpl::init(SimulationContextApi * context)
{
	auto _context = static_cast<SimulationContextGpuImpl*>(context);
	_worker = _context->getGpuThreadController()->getGpuWorker();
	connect(_worker, &GpuWorker::dataReadyToRetrieve, this, &SimulationAccessGpuImpl::dataReadyToRetrieveFromGpu);
}

void SimulationAccessGpuImpl::updateData(DataDescription const & desc)
{
}

void SimulationAccessGpuImpl::requireData(IntRect rect, ResolveDescription const & resolveDesc)
{
	_dataRequired = true;
	_requiredRect = rect;
	_resolveDesc = resolveDesc;

}

void SimulationAccessGpuImpl::requireImage(IntRect rect, QImage * target)
{
	_imageRequired = true;
	_requiredRect = rect;
	_requiredImage = target;

	_worker->requireData();
}

DataDescription const & SimulationAccessGpuImpl::retrieveData()
{
	return _dataCollected;
}

void SimulationAccessGpuImpl::dataReadyToRetrieveFromGpu()
{
	auto cudaData = _worker->retrieveData();
	if (_imageRequired) {
		//todo: create image

		Q_EMIT imageReadyToRetrieve();
	}
}

