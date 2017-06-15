#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <helper_cuda.h>

#include <stdio.h>
#include <functional>

#include "CudaBase.cuh"
#include "CudaConstants.cuh"
#include "CudaShared.cuh"
#include "CudaMovement.cuh"
#include "CudaHostHelper.cuh"

namespace {
	cudaStream_t cudaStream;
	CudaData *cudaData;
}

void createCluster(CudaCellCluster* cluster, float2 pos, float2 vel, float angle, float angVel, int2 clusterSize, int2 const &size)
{
	cluster->pos = pos;
	cluster->vel = vel;
	cluster->angle = angle;
	cluster->angularVel = angVel;
	cluster->numCells = clusterSize.x * clusterSize.y;
	cluster->cells = cudaData->cellsAC1.getArray(clusterSize.x*clusterSize.y);

	for (int x = 0; x < clusterSize.x; ++x) {
		for (int y = 0; y < clusterSize.y; ++y) {
			CudaCell *cell = &cluster->cells[x + y*clusterSize.x];
			cell->relPos = { static_cast<float>(x), static_cast<float>(y) };
			cell->cluster = cluster;
			cell->nextTimestep = nullptr;
			cell->protectionCounter = 0;
			cell->numConnections = 0;
			if (x > 0) {
				cell->connections[cell->numConnections++] = &cluster->cells[x - 1 + y * clusterSize.x];
			}
			if (y > 0) {
				cell->connections[cell->numConnections++] = &cluster->cells[x + (y - 1) * clusterSize.x];
			}
			if (x < clusterSize.x - 1) {
				cell->connections[cell->numConnections++] = &cluster->cells[x + 1 + y * clusterSize.x];
			}
			if (y < clusterSize.y - 1) {
				cell->connections[cell->numConnections++] = &cluster->cells[x + (y + 1) * clusterSize.x];
			}
		}
	}
	centerCluster(cluster);
	updateAbsPos(cluster);
	updateAngularMass(cluster);
}

void cudaInit(int2 const &size)
{
	cudaStreamCreate(&cudaStream);
	cudaSetDevice(0);

	cudaData = new CudaData();
	
	cudaData->size = size;
	cudaData->clustersAC1 = ArrayController<CudaCellCluster>(static_cast<int>(NUM_CLUSTERS * 1.1));
	cudaData->clustersAC2 = ArrayController<CudaCellCluster>(static_cast<int>(NUM_CLUSTERS * 1.1));
	cudaData->cellsAC1 = ArrayController<CudaCell>(static_cast<int>(NUM_CLUSTERS * 30 * 30 * 1.1));
	cudaData->cellsAC2 = ArrayController<CudaCell>(static_cast<int>(NUM_CLUSTERS * 30 * 30 * 1.1));
	size_t mapSize = size.x * size.y * sizeof(CudaCell*);
	cudaMallocManaged(&cudaData->map1, mapSize);
	cudaMallocManaged(&cudaData->map2, mapSize);
	checkCudaErrors(cudaGetLastError());
	for (int i = 0; i < size.x * size.y; ++i) {
		cudaData->map1[i] = nullptr;
		cudaData->map2[i] = nullptr;
	}


	auto clusters = cudaData->clustersAC1.getArray(NUM_CLUSTERS);

	for (int i = 0; i < NUM_CLUSTERS; ++i) {
		createCluster(&clusters[i], { 0.0f, 0.0f }, { random(0.5f) - 0.25f, random(0.5f) - 0.25f }, random(360.0f), random(0.2f) - 0.1f, { rand() % 20 + 1, rand() % 20 + 1 }, size);
		do {
			clusters[i].pos = { random(static_cast<float>(size.x)), random(static_cast<float>(size.y)) };
			centerCluster(&clusters[i]);
			updateAbsPos(&clusters[i]);

		} while (!isClusterPositionFree(&clusters[i], cudaData));

		drawClusterToMap(&clusters[i], cudaData);
		updateAngularMass(&clusters[i]);
	}

}


void cudaCalcNextTimestep()
{

	movement_Kernel <<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK, 0, cudaStream>>> (*cudaData);
	cudaDeviceSynchronize();
	clearOldMap_Kernel <<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK, 0, cudaStream>>> (*cudaData);
	cudaDeviceSynchronize();

	checkCudaErrors(cudaGetLastError());
	
	cudaData->swapData();
	cudaData->prepareTargetData();
}

void cudaGetSimulationDataRef(int& numClusters, CudaCellCluster*& clusters)
{
	numClusters = cudaData->clustersAC1.getNumEntries();
	clusters = cudaData->clustersAC1.getEntireArray();
}


void cudaShutdown()
{
	cudaDeviceSynchronize();
	cudaDeviceReset();

	cudaData->cellsAC1.free();
	cudaData->clustersAC1.free();
	cudaData->cellsAC2.free();
	cudaData->clustersAC2.free();

	cudaFree(cudaData->map1);
	cudaFree(cudaData->map2);

	delete cudaData;
}

