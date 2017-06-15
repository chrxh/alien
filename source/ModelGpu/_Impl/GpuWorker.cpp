#include <functional>
#include <QThread>
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

void GpuWorker::getData(IntRect const & rect, ResolveDescription const & resolveDesc, DataDescription & result)
{
	int numCLusters;
	CudaCellCluster* clusters;
	result.clear();
	cudaGetSimulationDataRef(numCLusters, clusters);
	for (int i = 0; i < numCLusters; ++i) {
		CellClusterDescription clusterDesc;
		CudaCellCluster temp = clusters[i];
		if (rect.isContained({ static_cast<int>(clusters[i].pos.x), static_cast<int>(clusters[i].pos.y) }))
		for (int j = 0; j < clusters[i].numCells; ++j) {
			auto pos = clusters[i].cells[j].absPos;
			clusterDesc.addCell(CellDescription().setPos({ static_cast<float>(pos.x), static_cast<float>(pos.y) }).setMetadata(CellMetadata()).setEnergy(100.0f));
		}
		result.addCellCluster(clusterDesc);
	}
}

const QColor UNIVERSE_COLOR(0x00, 0x00, 0x1b);

void GpuWorker::getImage(IntRect const & rect, QImage * image)
{
	image->fill(UNIVERSE_COLOR);

	int numCLusters;
	CudaCellCluster* clusters;
	cudaGetSimulationDataRef(numCLusters, clusters);
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

void GpuWorker::calculateTimestep()
{
	cudaCalcNextTimestep();
//	QThread::msleep(20);
	Q_EMIT timestepCalculated();
}
