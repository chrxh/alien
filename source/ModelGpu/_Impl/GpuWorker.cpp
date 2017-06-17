#include <functional>
#include <QImage>

#include "Model/SpaceMetricApi.h"
#include "ModelGpu/_Impl/Cuda/CudaShared.cuh"

#include "GpuWorker.h"

GpuWorker::~GpuWorker()
{
	cudaShutdown();
}

void GpuWorker::init(SpaceMetricApi* metric)
{
	_metric = metric;
	auto size = metric->getSize();
	cudaInit({ size.x, size.y });
}

void GpuWorker::requireData()
{
	_requireData = true;
}

CudaDataForAccess GpuWorker::lockAndRetrieveData()
{
	return _cudaData;
}

void GpuWorker::unlockData()
{
	_lockData = false;
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

bool GpuWorker::isSimulationRunning()
{
	return _simRunning;
}

void GpuWorker::setFlagStopAfterNextTimestep(bool value)
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

void GpuWorker::runSimulation()
{
	_simRunning = true;
	do {
		cudaCalcNextTimestep();
		Q_EMIT timestepCalculated();
		if (_requireData && !_lockData) {
			_lockData = true;	//TODO: mit mutex.trylock?
			_cudaData = cudaGetData();
			_requireData = false;
			Q_EMIT dataReadyToRetrieve();
		}
	} while (!_stopAfterNextTimestep);
	_simRunning = false;
}
