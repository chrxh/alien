#include <functional>
#include <QThread>

#include "Model/SpaceMetricApi.h"
#include "ModelGpu/_Impl/Cuda/CudaShared.cuh"

#include "GpuWorker.h"

GpuWorker::~GpuWorker()
{
	end_Cuda();
}

void GpuWorker::init(SpaceMetricApi* metric)
{
	auto size = metric->getSize();
	init_Cuda({ size.x, size.y });
}

void GpuWorker::getData(IntRect const & rect, ResolveDescription const & resolveDesc, DataDescription & result)
{
	int numCLusters;
	ClusterCuda* clusters;
	result.clear();
	getDataRef_Cuda(numCLusters, clusters);
	for (int i = 0; i < numCLusters; ++i) {
		CellClusterDescription clusterDesc;
		ClusterCuda temp = clusters[i];
		if (rect.isContained({ static_cast<int>(clusters[i].pos.x), static_cast<int>(clusters[i].pos.y) }))
		for (int j = 0; j < clusters[i].numCells; ++j) {
			auto pos = clusters[i].cells[j].absPos;
			clusterDesc.addCell(CellDescription().setPos({ static_cast<float>(pos.x), static_cast<float>(pos.y) }).setMetadata(CellMetadata()).setEnergy(100.0f));
		}
		result.addCellCluster(clusterDesc);
	}
}

void GpuWorker::calculateTimestep()
{
	calcNextTimestep_Cuda();
	QThread::msleep(20);
	Q_EMIT timestepCalculated();
}
