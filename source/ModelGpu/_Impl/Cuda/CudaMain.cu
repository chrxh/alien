#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <helper_cuda.h>

#include <stdio.h>
#include <functional>

#include "CudaBase.cuh"
#include "CudaConstants.cuh"
#include "CudaShared.cuh"
#include "CudaMovement.cuh"
#include "CudaHostHelperFunctions.cuh"

cudaStream_t cudaStream;
CudaData cudaData;

void createCluster(ClusterCuda* cluster, double2 pos, double2 vel, double angle, double angVel, int2 clusterSize, int2 const &size)
{
	cluster->pos = pos;
	cluster->vel = vel;
	cluster->angle = angle;
	cluster->angularVel = angVel;
	cluster->numCells = clusterSize.x * clusterSize.y;
	cluster->cells = cudaData.cellsAC1.getArray(clusterSize.x*clusterSize.y);

	for (int x = 0; x < clusterSize.x; ++x) {
		for (int y = 0; y < clusterSize.y; ++y) {
			CellCuda *cell = &cluster->cells[x + y*clusterSize.x];
			cell->relPos = { static_cast<double>(x), static_cast<double>(y) };
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

void init_Cuda(int2 size)
{
	cudaStreamCreate(&cudaStream);
	cudaSetDevice(0);

	cudaData.size = size;
	size_t mapSize = size.x * size.y * sizeof(CellCuda*) * LAYERS;
	cudaMallocManaged(&cudaData.map1, mapSize);
	cudaMallocManaged(&cudaData.map2, mapSize);
	for (int i = 0; i < size.x * size.y * LAYERS; ++i) {
		cudaData.map1[i] = nullptr;
		cudaData.map2[i] = nullptr;
	}
	cudaData.clustersAC1 = ArrayController<ClusterCuda>(static_cast<int>(NUM_CLUSTERS * 1.1));
	cudaData.clustersAC2 = ArrayController<ClusterCuda>(static_cast<int>(NUM_CLUSTERS * 1.1));
	cudaData.cellsAC1 = ArrayController<CellCuda>(static_cast<int>(NUM_CLUSTERS * 400 * 1.1));
	cudaData.cellsAC2 = ArrayController<CellCuda>(static_cast<int>(NUM_CLUSTERS * 400 * 1.1));

	auto clusters = cudaData.clustersAC1.getArray(NUM_CLUSTERS);

/*
	createCluster(&clusters[0], { 1500.5, 1200 }, { 0.001, 0 }, 90, 0, 64, size);
	createCluster(&clusters[1], { 1550.5, 1175 }, { -0.2, 0 }, 0, 0.0, 64, size);
	drawClusterToMap(&clusters[0], &cudaData);
	drawClusterToMap(&clusters[1], &cudaData);
*/


	for (int i = 0; i < NUM_CLUSTERS; ++i) {
		createCluster(&clusters[i], { random(size.x), random(size.y) }, { random(0.5f) - 0.25f, random(0.5) - 0.25f }, random(360.0f), random(0.2f) - 0.1f, { rand() % 20 + 1, rand() % 20 + 1 }, size);
		do {
			clusters[i].pos = { random(size.x), random(size.y) };
			centerCluster(&clusters[i]);
			updateAbsPos(&clusters[i]);

		} while (!isClusterPositionFree(&clusters[i], &cudaData));

		drawClusterToMap(&clusters[i], &cudaData);
		updateAngularMass(&clusters[i]);
	}

}


void calcNextTimestep_Cuda()
{

	movement_Kernel <<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK, 0, cudaStream>>> (cudaData);
	cudaDeviceSynchronize();
	clearOldMap_Kernel <<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK, 0, cudaStream >>> (cudaData);
	cudaDeviceSynchronize();

	checkCudaErrors(cudaGetLastError());
	//****
/*
	if (abs(cudaData.clustersAC2.getEntireArray()[0].vel.x+0.2) > 0.0002 && abs(cudaData.clustersAC2.getEntireArray()[1].vel.x + 0.2) > 0.0002) {
		ClusterCuda temp1 = cudaData.clustersAC2.getEntireArray()[0];
		ClusterCuda temp2 = cudaData.clustersAC2.getEntireArray()[1];
		int dummy = 0;
	}
*/
	//****
	swap(cudaData.clustersAC1, cudaData.clustersAC2);
	swap(cudaData.cellsAC1, cudaData.cellsAC2);
	swap(cudaData.map1, cudaData.map2);
	cudaData.clustersAC2.reset();
	cudaData.cellsAC2.reset();
}

void getDataRef_Cuda(int& numClusters, ClusterCuda*& clusters)
{
	numClusters = cudaData.clustersAC1.getNumEntries();
	clusters = cudaData.clustersAC1.getEntireArray();
}


void end_Cuda()
{
	cudaDeviceSynchronize();
	cudaDeviceReset();

	cudaData.cellsAC1.free();
	cudaData.clustersAC1.free();
	cudaData.cellsAC2.free();
	cudaData.clustersAC2.free();
	cudaFree(cudaData.map1);
	cudaFree(cudaData.map2);
}

