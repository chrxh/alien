#include <QImage>

#include "ModelBasic/SpaceProperties.h"

#include "GpuWorker.h"
#include "ThreadController.h"
#include "SimulationContextGpuImpl.h"
#include "SimulationAccessGpuImpl.h"

SimulationAccessGpuImpl::~SimulationAccessGpuImpl()
{
}

void SimulationAccessGpuImpl::init(SimulationControllerGpu* controller)
{
	_context = static_cast<SimulationContextGpuImpl*>(context);
	auto worker = _context->getGpuThreadController()->getGpuWorker();
	connect(worker, &GpuWorker::dataReadyToRetrieve, this, &SimulationAccessGpuImpl::dataReadyToRetrieveFromGpu);
}

void SimulationAccessGpuImpl::clear()
{
}

void SimulationAccessGpuImpl::updateData(DataChangeDescription const & desc)
{
}

void SimulationAccessGpuImpl::requireData(IntRect rect, ResolveDescription const & resolveDesc)
{
	_dataRequired = true;
	_requiredRect = rect;
	_resolveDesc = resolveDesc;

	auto worker = _context->getGpuThreadController()->getGpuWorker();
	worker->requireData();
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

	if (_imageRequired) {
		_imageRequired = false;
		createImage();
		Q_EMIT imageReady();

	}

	if (_dataRequired) {
		_dataRequired = false;
		createData();
		Q_EMIT dataReadyToRetrieve();
	}
}

void SimulationAccessGpuImpl::createImage()
{
	auto metric = _context->getSpaceProperties();
	auto worker = _context->getGpuThreadController()->getGpuWorker();
	DataForAccess cudaData = worker->retrieveData();

	_requiredImage->fill(QColor(0x00, 0x00, 0x1b));

	worker->lockData();

	for (int i = 0; i < cudaData.numParticles; ++i) {
		ParticleData& particle = cudaData.particles[i];
		float2& pos = particle.pos;
		IntVector2D intPos = { static_cast<int>(pos.x), static_cast<int>(pos.y) };
		if (_requiredRect.isContained(intPos)) {
			metric->correctPosition(intPos);
			_requiredImage->setPixel(intPos.x, intPos.y, 0x902020);
		}
	}

	for (int i = 0; i < cudaData.numCells; ++i) {
		CellData& cell = cudaData.cells[i];
		float2& pos = cell.absPos;
		IntVector2D intPos = { static_cast<int>(pos.x), static_cast<int>(pos.y) };
		if (_requiredRect.isContained(intPos)) {
			metric->correctPosition(intPos);
			_requiredImage->setPixel(intPos.x, intPos.y, 0xFF);
		}
	}

	worker->unlockData();
}

void SimulationAccessGpuImpl::createData()
{
	_dataCollected.clear();
	auto worker = _context->getGpuThreadController()->getGpuWorker();
	DataForAccess cudaData = worker->retrieveData();

	worker->lockData();
	worker->ptrCorrectionForRetrievedData();

	for (int i = 0; i < cudaData.numClusters; ++i) {
		ClusterDescription clusterDesc;
		if (_requiredRect.isContained({ static_cast<int>(cudaData.clusters[i].pos.x), static_cast<int>(cudaData.clusters[i].pos.y) }))
			for (int j = 0; j < cudaData.clusters[i].numCells; ++j) {
				auto pos = cudaData.clusters[i].cells[j].absPos;
				auto id = cudaData.clusters[i].cells[j].id;
				clusterDesc.addCell(CellDescription().setPos({ pos.x, pos.y }).setMetadata(CellMetadata()).setEnergy(100.0f).setId(id));
			}
		_dataCollected.addCluster(clusterDesc);
	}

	worker->unlockData();
}

