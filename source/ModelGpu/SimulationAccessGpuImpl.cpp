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
	const int NumDataTOs = 5;
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
	_lastDataRect = { { 0,0 }, size };
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
	IntRect rect = cudaWorker->isSimulationRunning() ? IntRect{ { 0, 0 }, size } : _lastDataRect;

	auto updateDescCorrected = updateDesc;
	metricCorrection(updateDescCorrected);

	CudaJob job = boost::make_shared<_GetDataForUpdateJob>(getObjectId(), _lastDataRect, _dataTOCache.getDataTO(), updateDescCorrected);
	cudaWorker->addJob(job);
	_updateInProgress = true;
}

void SimulationAccessGpuImpl::requireData(IntRect rect, ResolveDescription const & resolveDesc)
{
	auto worker = _context->getCudaController()->getCudaWorker();
	CudaJob job = boost::make_shared<_GetDataForEditJob>(getObjectId(), rect, _dataTOCache.getDataTO());
	if (!_updateInProgress) {
		worker->addJob(job);
	}
	else {
		_waitingJobs.push_back(job);
	}
}

void SimulationAccessGpuImpl::requireImage(IntRect rect, QImage * target)
{
	auto worker = _context->getCudaController()->getCudaWorker();
	CudaJob job = boost::make_shared<_GetDataForImageJob>(getObjectId(), rect, _dataTOCache.getDataTO(), target);
	if (!_updateInProgress) {
		worker->addJob(job);
	}
	else {
		_waitingJobs.push_back(job);
	}
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
			updateDataToGpu(dataToUpdateTO, getDataForUpdateJob->getRect(), getDataForUpdateJob->getUpdateDescription());
			Q_EMIT dataUpdated();
		}

		if (auto const& getDataForImageJob = boost::dynamic_pointer_cast<_GetDataForImageJob>(job)) {
			auto dataTO = getDataForImageJob->getDataTO();
			createImageFromGpuModel(dataTO, getDataForImageJob->getRect(), getDataForImageJob->getTargetImage());
			_dataTOCache.releaseDataTO(dataTO);
			Q_EMIT imageReady();
		}

		if (auto const& getDataForEditJob = boost::dynamic_pointer_cast<_GetDataForEditJob>(job)) {
			auto dataTO = getDataForEditJob->getDataTO();
			createDataFromGpuModel(dataTO, getDataForEditJob->getRect());
			_dataTOCache.releaseDataTO(dataTO);
			Q_EMIT dataReadyToRetrieve();
		}

		if (auto const& setDataJob = boost::dynamic_pointer_cast<_SetDataJob>(job)) {
			_dataTOCache.releaseDataTO(setDataJob->getDataTO());
			_updateInProgress = false;
			for (auto const& job : _waitingJobs) {
				worker->addJob(job);
			}
			_waitingJobs.clear();
		}
	}
}

void SimulationAccessGpuImpl::updateDataToGpu(DataAccessTO dataToUpdateTO, IntRect const& rect, DataChangeDescription const& updateDesc)
{
	DataConverter converter(dataToUpdateTO, _numberGen, _context->getSimulationParameters());
	converter.updateData(updateDesc);

	auto cudaWorker = _context->getCudaController()->getCudaWorker();
	CudaJob job = boost::make_shared<_SetDataJob>(getObjectId(), true, rect, dataToUpdateTO);
	cudaWorker->addJob(job);
}

void SimulationAccessGpuImpl::createImageFromGpuModel(DataAccessTO const& dataTO, IntRect const& rect, QImage* targetImage)
{
	auto space = _context->getSpaceProperties();
	auto worker = _context->getCudaController()->getCudaWorker();
	EntityRenderer renderer(targetImage, space);

	auto truncatedRect = rect;
	space->truncateRect(truncatedRect);
	//	EntityRenderer::fillRect(targetImage, truncatedRect);
	targetImage->fill(QColor(0, 0, 0x1b));


	for (int i = 0; i < *dataTO.numParticles; ++i) {
		ParticleAccessTO& particle = dataTO.particles[i];
		float2& pos = particle.pos;
		IntVector2D intPos = { static_cast<int>(pos.x), static_cast<int>(pos.y) };
		renderer.renderParticle(intPos, particle.energy);
	}

	for (int i = 0; i < *dataTO.numCells; ++i) {
		CellAccessTO& cell = dataTO.cells[i];
		float2 const& pos = cell.pos;
		IntVector2D intPos = { static_cast<int>(pos.x), static_cast<int>(pos.y) };
		renderer.renderCell(intPos, 0, cell.energy);
	}

	for (int i = 0; i < *dataTO.numTokens; ++i) {
		TokenAccessTO const& token = dataTO.tokens[i];
		CellAccessTO const& cell = dataTO.cells[token.cellIndex];
		IntVector2D pos = { static_cast<int>(cell.pos.x), static_cast<int>(cell.pos.y) };
		renderer.renderToken(pos);
	}
}

void SimulationAccessGpuImpl::createDataFromGpuModel(DataAccessTO dataTO, IntRect const& rect)
{
	_lastDataRect = rect;

	DataConverter converter(dataTO, _numberGen, _context->getSimulationParameters());
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
		dataTO.numTokens = new int;
		dataTO.clusters = new ClusterAccessTO[MAX_CLUSTERS];
		dataTO.cells = new CellAccessTO[MAX_CELLS];
		dataTO.particles = new ParticleAccessTO[MAX_PARTICLES];
		dataTO.tokens = new TokenAccessTO[MAX_TOKENS];
		_freeDataTOs.push_back(dataTO);
	}
}

SimulationAccessGpuImpl::DataTOCache::~DataTOCache()
{
	for (DataAccessTO const& dataTO : _freeDataTOs) {
		delete dataTO.numClusters;
		delete dataTO.numCells;
		delete dataTO.numParticles;
		delete dataTO.numTokens;
		delete[] dataTO.clusters;
		delete[] dataTO.cells;
		delete[] dataTO.particles;
		delete[] dataTO.tokens;
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
		return usedDataTO == dataTO;
	});
	if (usedDataTO != _usedDataTOs.end()) {
		_freeDataTOs.push_back(*usedDataTO);
		_usedDataTOs.erase(usedDataTO);
	}
}
