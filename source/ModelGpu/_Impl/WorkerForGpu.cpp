#include <functional>
#include <QImage>

#include "Model/SpaceMetricApi.h"
#include "ModelGpu/_Impl/Cuda/CudaInterface.cuh"

#include "WorkerForGpu.h"

WorkerForGpu::~WorkerForGpu()
{
	cudaShutdown();
}

void WorkerForGpu::init(SpaceMetricApi* metric)
{
	_metric = metric;
	auto size = metric->getSize();
	cudaInit({ size.x, size.y });
}

void WorkerForGpu::requireData()
{
	_requireData = true;
}

DataForAccess WorkerForGpu::retrieveData()
{
	return _cudaData;
}

void WorkerForGpu::lockData()
{
	_mutex.lock();
}

void WorkerForGpu::unlockData()
{
	_mutex.unlock();
}

/*
void GpuWorker::getData(IntRect const & rect, ResolveDescription const & resolveDesc, DataDescription & result)
{
	result.clear();
	CudaData data = cudaGetDataRef();
	
	for (int i = 0; i < data.numClusters; ++i) {
		CellClusterDescription clusterDesc;
		CudaCellCluster temp = data.clusters[i];
		if (rect.isContained({ static_cast<int>(data.clusters[i].pos.x), static_cast<int>(data.clusters[i].pos.y) }))
		for (int j = 0; j < data.clusters[i].numCells; ++j) {
			auto pos = data.clusters[i].cells[j].absPos;
			clusterDesc.addCell(CellDescription().setPos({ static_cast<float>(pos.x), static_cast<float>(pos.y) }).setMetadata(CellMetadata()).setEnergy(100.0f));
		}
		result.addCellCluster(clusterDesc);
	}
}
*/

bool WorkerForGpu::isSimulationRunning()
{
	return _simRunning;
}

void WorkerForGpu::setFlagStopAfterNextTimestep(bool value)
{
	_stopAfterNextTimestep = value;
}

/*
const QColor UNIVERSE_COLOR(0x00, 0x00, 0x1b);

void GpuWorker::getImage(IntRect const & rect, QImage * image)
{
	image->fill(UNIVERSE_COLOR);

	int numCLusters;
	CudaCellCluster* clusters;
	cudaGetClustersRef(numCLusters, clusters);
	for (int i = 0; i < numCLusters; ++i) {
		CellClusterDescription clusterDesc;
		CudaCellCluster temp = clusters[i];
		if (rect.isContained({ static_cast<int>(clusters[i].pos.x), static_cast<int>(clusters[i].pos.y) }))
			for (int j = 0; j < clusters[i].numCells; ++j) {
				float2 pos = clusters[i].cells[j].absPos;
				IntVector2D intPos = { static_cast<int>(pos.x), static_cast<int>(pos.y) };
				 _metric->correctPosition(intPos);
				 image->setPixel(intPos.x, intPos.y, 0xFF);
			}
	}
}
*/

void WorkerForGpu::runSimulation()
{
	_simRunning = true;
	do {
		cudaCalcNextTimestep();
		Q_EMIT timestepCalculated();
		if (_requireData && _mutex.try_lock()) {
			_cudaData = cudaGetData();
			_requireData = false;
			_mutex.unlock();
			Q_EMIT dataReadyToRetrieve();
		}
	} while (!_stopAfterNextTimestep);
	_simRunning = false;
}
