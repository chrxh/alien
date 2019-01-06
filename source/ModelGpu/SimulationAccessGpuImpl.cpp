#include <QImage>

#include "ModelBasic/SpaceProperties.h"
#include "ModelBasic/EntityRenderer.h"

#include "CudaWorker.h"
#include "ThreadController.h"
#include "SimulationContextGpuImpl.h"
#include "SimulationAccessGpuImpl.h"
#include "SimulationControllerGpu.h"
#include "DataConverter.h"

SimulationAccessGpuImpl::~SimulationAccessGpuImpl()
{
}

void SimulationAccessGpuImpl::init(SimulationControllerGpu* controller)
{
	_context = static_cast<SimulationContextGpuImpl*>(controller->getContext());
	_numberGen = _context->getNumberGenerator();
	auto cudaBridge = _context->getGpuThreadController()->getCudaBridge();
	connect(cudaBridge, &CudaWorker::dataObtained, this, &SimulationAccessGpuImpl::dataRequiredFromGpu, Qt::QueuedConnection);
}

void SimulationAccessGpuImpl::clear()
{
}

void SimulationAccessGpuImpl::updateData(DataChangeDescription const & desc)
{
	_dataToUpdate.clusters.insert(_dataToUpdate.clusters.end(), desc.clusters.begin(), desc.clusters.end());
	_dataToUpdate.particles.insert(_dataToUpdate.particles.end(), desc.particles.begin(), desc.particles.end());

	auto cudaBridge = _context->getGpuThreadController()->getCudaBridge();
	cudaBridge->requireData();
}

void SimulationAccessGpuImpl::requireData(IntRect rect, ResolveDescription const & resolveDesc)
{
	_dataDescRequired = true;
	_requiredRect = rect;
	_resolveDesc = resolveDesc;

	auto cudaBridge = _context->getGpuThreadController()->getCudaBridge();
	cudaBridge->requireData();
}

void SimulationAccessGpuImpl::requireImage(IntRect rect, QImage * target)
{
	_imageRequired = true;
	_requiredRect = rect;
	_requiredImage = target;

	auto cudaBridge = _context->getGpuThreadController()->getCudaBridge();
	cudaBridge->requireData();
}

DataDescription const & SimulationAccessGpuImpl::retrieveData()
{
	return _dataCollected;
}

void SimulationAccessGpuImpl::dataRequiredFromGpu()
{
	if (!_dataToUpdate.empty()) {
		updateDataToGpuModel();
		Q_EMIT dataUpdated();
	}

	if (_imageRequired) {
		_imageRequired = false;
		createImageFromGpuModel();
		Q_EMIT imageReady();
	}

	if (_dataDescRequired) {
		_dataDescRequired = false;
		createDataFromGpuModel();
		Q_EMIT dataReadyToRetrieve();
	}
}

void SimulationAccessGpuImpl::updateDataToGpuModel()
{
	auto cudaBridge = _context->getGpuThreadController()->getCudaBridge();

	cudaBridge->lockData();
	SimulationDataForAccess& cudaData = cudaBridge->retrieveData();

	DataConverter converter(cudaData, _numberGen);
	converter.updateData(_dataToUpdate);

	cudaBridge->updateData();
	cudaBridge->unlockData();
	_dataToUpdate.clear();
}

void SimulationAccessGpuImpl::createImageFromGpuModel()
{
	auto spaceProp = _context->getSpaceProperties();
	auto cudaBridge = _context->getGpuThreadController()->getCudaBridge();

	_requiredImage->fill(QColor(0x00, 0x00, 0x1b));

	cudaBridge->lockData();
	SimulationDataForAccess cudaData = cudaBridge->retrieveData();

	for (int i = 0; i < cudaData.numParticles; ++i) {
		ParticleData& particle = cudaData.particles[i];
		float2& pos = particle.pos;
		IntVector2D intPos = { static_cast<int>(pos.x), static_cast<int>(pos.y) };
		if (_requiredRect.isContained(intPos)) {
			spaceProp->correctPosition(intPos);
			_requiredImage->setPixel(intPos.x, intPos.y, EntityRenderer::calcParticleColor(particle.energy));
		}
	}

	for (int i = 0; i < cudaData.numCells; ++i) {
		CellData& cell = cudaData.cells[i];
		float2& pos = cell.absPos;
		IntVector2D intPos = { static_cast<int>(pos.x), static_cast<int>(pos.y) };
		if (_requiredRect.isContained(intPos)) {
			spaceProp->correctPosition(intPos);
			_requiredImage->setPixel(intPos.x, intPos.y, EntityRenderer::calcCellColor(0, 0, cell.energy));
		}
	}

	cudaBridge->unlockData();
}

void SimulationAccessGpuImpl::createDataFromGpuModel()
{
	_dataCollected.clear();
	auto cudaBridge = _context->getGpuThreadController()->getCudaBridge();

	cudaBridge->lockData();
	SimulationDataForAccess cudaData = cudaBridge->retrieveData();
	DataConverter converter(cudaData, _numberGen);
	_dataCollected = converter.getDataDescription(_requiredRect);

	cudaBridge->unlockData();
}

