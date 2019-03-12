#include <sstream>
#include <QImage>

#include "ModelBasic/SpaceProperties.h"
#include "ModelBasic/EntityRenderer.h"

#include "CudaWorker.h"
#include "CudaController.h"
#include "SimulationContextGpuImpl.h"
#include "SimulationAccessGpuImpl.h"
#include "SimulationControllerGpu.h"
#include "CudaJob.h"
#include "DataConverter.h"

namespace
{
	const string SimulationAccessGpuId = "SimulationAccessGpuId";
	const int NumDataTOs = 4;
}

SimulationAccessGpuImpl::SimulationAccessGpuImpl(QObject* parent /*= nullptr*/)
	: SimulationAccessGpu(parent)
{
}

SimulationAccessGpuImpl::~SimulationAccessGpuImpl()
{
}

void SimulationAccessGpuImpl::init(SimulationControllerGpu* controller)
{
	_context = static_cast<SimulationContextGpuImpl*>(controller->getContext());
	_numberGen = _context->getNumberGenerator();
	auto worker = _context->getCudaController()->getCudaWorker();
	auto size = _context->getSpaceProperties()->getSize();
	_lastRect = { { 0,0 }, size };
	for (auto const& connection : _connections) {
		QObject::disconnect(connection);
	}
	_connections.push_back(connect(worker, &CudaWorker::jobsFinished, this, &SimulationAccessGpuImpl::jobsFinished, Qt::QueuedConnection));
}

void SimulationAccessGpuImpl::clear()
{
}

void SimulationAccessGpuImpl::updateData(DataChangeDescription const& updateDesc)
{
	auto cudaWorker = _context->getCudaController()->getCudaWorker();

	//heuristic for determining rect
	auto size = _context->getSpaceProperties()->getSize();
	IntRect rect = cudaWorker->isSimulationRunning() ? IntRect{ { 0, 0 }, size } : _lastRect;

	auto updateDescCorrected = updateDesc;
	metricCorrection(updateDescCorrected);

	CudaJob job = boost::make_shared<_GetDataForUpdateJob>(getObjectId(), _lastRect, _dataTOCache.getDataTO(), updateDescCorrected);
	cudaWorker->addJob(job);
}

void SimulationAccessGpuImpl::requireData(IntRect rect, ResolveDescription const & resolveDesc)
{
	_lastRect = rect;
	auto cudaWorker = _context->getCudaController()->getCudaWorker();
	CudaJob job = boost::make_shared<_GetDataForEditJob>(getObjectId(), rect, _dataTOCache.getDataTO());
	cudaWorker->addJob(job);
}

void SimulationAccessGpuImpl::requireImage(IntRect rect, QImage * target)
{
	auto cudaWorker = _context->getCudaController()->getCudaWorker();
	CudaJob job = boost::make_shared<_GetDataForImageJob>(getObjectId(), rect, _dataTOCache.getDataTO(), target);
	cudaWorker->addJob(job);
}

DataDescription const & SimulationAccessGpuImpl::retrieveData()
{
	return _dataCollected;
}

void SimulationAccessGpuImpl::jobsFinished()
{
	auto worker = _context->getCudaController()->getCudaWorker();
	auto finishedJobs = worker->getFinishedJobs(getObjectId());
	for (auto const& job : finishedJobs) {

		if (auto const& getDataForUpdateJob = boost::dynamic_pointer_cast<_GetDataForUpdateJob>(job)) {
			auto dataToUpdateTO = getDataForUpdateJob->getDataTO();
			updateDataToGpu(dataToUpdateTO, getDataForUpdateJob->getUpdateDescription());
			Q_EMIT dataUpdated();
		}

		if (auto const& getDataForImageJob = boost::dynamic_pointer_cast<_GetDataForImageJob>(job)) {
			auto dataTO = getDataForImageJob->getDataTO();
			createImageFromGpuModel(dataTO, getDataForImageJob->getTargetImage());
			_dataTOCache.releaseDataTO(dataTO);
			Q_EMIT imageReady();
		}

		if (auto const& getDataForEditJob = boost::dynamic_pointer_cast<_GetDataForEditJob>(job)) {
			auto dataTO = getDataForEditJob->getDataTO();
			createDataFromGpuModel(dataTO);
			_dataTOCache.releaseDataTO(dataTO);
			Q_EMIT dataReadyToRetrieve();
		}

		if (auto const& setDataJob = boost::dynamic_pointer_cast<_SetDataJob>(job)) {
			_dataTOCache.releaseDataTO(setDataJob->getDataTO());
		}
	}
}

void SimulationAccessGpuImpl::updateDataToGpu(DataAccessTO dataToUpdateTO, DataChangeDescription const& updateDesc)
{
	DataConverter converter(dataToUpdateTO, _numberGen);
	converter.updateData(updateDesc);

	auto cudaWorker = _context->getCudaController()->getCudaWorker();
	CudaJob job = boost::make_shared<_SetDataJob>(getObjectId(), true, _lastRect, dataToUpdateTO);
	cudaWorker->addJob(job);
}

namespace
{
	void colorPixel(QImage* image, IntVector2D const& pos, QRgb const& color, int alpha)
	{
		QRgb const& origColor = image->pixel(pos.x, pos.y);

		int red = (qRed(color) * alpha + qRed(origColor) * (255 - alpha)) / 255;
		int green = (qGreen(color) * alpha + qGreen(origColor) * (255 - alpha)) / 255;
		int blue = (qBlue(color) * alpha + qBlue(origColor) * (255 - alpha)) / 255;
		image->setPixel(pos.x, pos.y, qRgb(red, green, blue));
	}
}
void SimulationAccessGpuImpl::createImageFromGpuModel(DataAccessTO const& dataTO, QImage* targetImage)
{
	auto space = _context->getSpaceProperties();
	auto cudaWorker = _context->getCudaController()->getCudaWorker();

	targetImage->fill(QColor(0x00, 0x00, 0x1b));

	for (int i = 0; i < *dataTO.numParticles; ++i) {
		ParticleAccessTO& particle = dataTO.particles[i];
		float2& pos = particle.pos;
		IntVector2D intPos = { static_cast<int>(pos.x), static_cast<int>(pos.y) };
		space->correctPosition(intPos);
		targetImage->setPixel(intPos.x, intPos.y, EntityRenderer::calcParticleColor(particle.energy));
	}

	for (int i = 0; i < *dataTO.numCells; ++i) {
		CellAccessTO& cell = dataTO.cells[i];
		float2 const& pos = cell.pos;
		IntVector2D intPos = { static_cast<int>(pos.x), static_cast<int>(pos.y) };
		space->correctPosition(intPos);
		uint32_t color = EntityRenderer::calcCellColor(0, 0, cell.energy);
		targetImage->setPixel(intPos.x, intPos.y, color);
		--intPos.x;
		space->correctPosition(intPos);
		colorPixel(targetImage, intPos, color, 0x60);
		intPos.x += 2;
		space->correctPosition(intPos);
		colorPixel(targetImage, intPos, color, 0x60);
		--intPos.x;
		--intPos.y;
		space->correctPosition(intPos);
		colorPixel(targetImage, intPos, color, 0x60);
		intPos.y += 2;
		space->correctPosition(intPos);
		colorPixel(targetImage, intPos, color, 0x60);
	}
}

void SimulationAccessGpuImpl::createDataFromGpuModel(DataAccessTO dataTO)
{
	DataConverter converter(dataTO, _numberGen);
	_dataCollected = converter.getDataDescription();
}

void SimulationAccessGpuImpl::metricCorrection(DataChangeDescription & data) const
{
	SpaceProperties* space = _context->getSpaceProperties();
	for (auto& cluster : data.clusters) {
		QVector2D origPos = cluster->pos.getValue();
		auto pos = origPos;
		space->correctPosition(pos);
		auto correctionDelta = pos - origPos;
		if (!correctionDelta.isNull()) {
			cluster->pos.setValue(pos);
		}
		for (auto& cell : cluster->cells) {
			cell->pos.setValue(cell->pos.getValue() + correctionDelta);
		}
	}
	for (auto& particle : data.particles) {
		QVector2D origPos = particle->pos.getValue();
		auto pos = origPos;
		space->correctPosition(pos);
		if (pos != origPos) {
			particle->pos.setValue(pos);
		}
	}
}

string SimulationAccessGpuImpl::getObjectId() const
{
	auto id = reinterpret_cast<long long>(this);
	std::stringstream stream;
	stream << SimulationAccessGpuId << id;
	return stream.str();
}

SimulationAccessGpuImpl::DataTOCache::DataTOCache()
{
	for (int i = 0; i < NumDataTOs; ++i) {
		DataAccessTO dataTO;
		dataTO.numClusters = new int;
		dataTO.numCells = new int;
		dataTO.numParticles = new int;
		dataTO.clusters = new ClusterAccessTO[MAX_CELLCLUSTERS];
		dataTO.cells = new CellAccessTO[MAX_CELLS];
		dataTO.particles = new ParticleAccessTO[MAX_PARTICLES];
		_freeDataTOs.push_back(dataTO);
	}
}

SimulationAccessGpuImpl::DataTOCache::~DataTOCache()
{
	for (DataAccessTO const& dataTO : _freeDataTOs) {
		delete dataTO.numClusters;
		delete dataTO.numCells;
		delete dataTO.numParticles;
		delete[] dataTO.clusters;
		delete[] dataTO.cells;
		delete[] dataTO.particles;
	}
}

DataAccessTO SimulationAccessGpuImpl::DataTOCache::getDataTO()
{
	DataAccessTO result;
	if (!_freeDataTOs.empty()) {
		result = *_freeDataTOs.begin();
		_freeDataTOs.erase(_freeDataTOs.begin());
		_usedDataTOs.push_back(result);
		return result;
	}
	result = *_usedDataTOs.begin();
	return result;
}

void SimulationAccessGpuImpl::DataTOCache::releaseDataTO(DataAccessTO const & dataTO)
{
	auto usedDataTO = std::find_if(_usedDataTOs.begin(), _usedDataTOs.end(), [&dataTO](DataAccessTO const& usedDataTO) {
		return usedDataTO.numClusters == dataTO.numClusters;
	});
	_freeDataTOs.push_back(*usedDataTO);
	_usedDataTOs.erase(usedDataTO);
}
