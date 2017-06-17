#include <QImage>

#include "Model/SpaceMetricApi.h"

#include "GpuWorker.h"
#include "GpuThreadController.h"
#include "SimulationContextGpuImpl.h"
#include "SimulationAccessGpuImpl.h"

SimulationAccessGpuImpl::~SimulationAccessGpuImpl()
{
}

void SimulationAccessGpuImpl::init(SimulationContextApi * context)
{
	_context = static_cast<SimulationContextGpuImpl*>(context);
	auto worker = _context->getGpuThreadController()->getGpuWorker();
	connect(worker, &GpuWorker::dataReadyToRetrieve, this, &SimulationAccessGpuImpl::dataReadyToRetrieveFromGpu);
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

	auto worker = _context->getGpuThreadController()->getGpuWorker();
	worker->requireData();
}

DataDescription const & SimulationAccessGpuImpl::retrieveData()
{
	return _dataCollected;
}

void SimulationAccessGpuImpl::dataReadyToRetrieveFromGpu()
{
	auto worker = _context->getGpuThreadController()->getGpuWorker();
	auto metric = _context->getSpaceMetric();

	if (_imageRequired) {
		_imageRequired = false;

		auto cudaData = worker->lockAndRetrieveData();

		_requiredImage->fill(QColor(0x00, 0x00, 0x1b));

		for (int i = 0; i < cudaData.numCells; ++i) {
			CudaCell& cell = cudaData.cells[i];
			float2 pos = cell.absPos;
			IntVector2D intPos = { static_cast<int>(pos.x), static_cast<int>(pos.y) };
			if (_requiredRect.isContained(intPos)) {
				metric->correctPosition(intPos);
				_requiredImage->setPixel(intPos.x, intPos.y, 0xFF);
			}
		}

		worker->unlockData();

		Q_EMIT imageReady();
	}
}

