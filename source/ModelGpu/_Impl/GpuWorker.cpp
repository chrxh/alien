#include <functional>

#include "GpuWorker.h"
#include "CudaFunctions.cuh"

GpuWorker::~GpuWorker()
{
	end_Cuda();
}

void GpuWorker::init()
{
	init_Cuda();
}

void GpuWorker::getData(IntRect const & rect, ResolveDescription const & resolveDesc, DataDescription & result) const
{
	int numCLusters;
	ClusterCuda* clusters;
	result.clear();
	getData_Cuda(numCLusters, clusters);
	for (int i = 0; i < numCLusters; ++i) {
		CellClusterDescription clusterDesc;
		for (int j = 0; j < clusters[i].numCells; ++j) {
			clusterDesc.addCell(CellDescription().setPos({ (float)clusters[i].pos.x, (float)clusters[i].pos.y }).setMetadata(CellMetadata()).setEnergy(100.0f));
		}
		result.addCellCluster(clusterDesc);
	}
}

void GpuWorker::calculateTimestep()
{
	calcNextTimestep_Cuda();
	Q_EMIT timestepCalculated();
}
