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

void createCluster(ClusterCuda* cluster, double2 pos, double2 vel, double angle, double angVel, int cellsPerCluster, int2 const &size)
{
	cluster->pos = pos;/*{ random(size.x), random(size.y) };*/
	cluster->vel = vel;/*{ random(1.0f) - 0.5f, random(1.0) - 0.5f };*/
	cluster->angle = angle;/*random(360.0f);*/
	cluster->angularVel = angVel;/*random(0.1f) - 0.2f;*/
	cluster->numCells = cellsPerCluster;
	cluster->cells = cudaData.cellsAC1.getArray(cellsPerCluster);

	for (int j = 0; j < cellsPerCluster; ++j) {
		CellCuda *cell = &cluster->cells[j];
		cell->relPos = { j - 31.5f, 0 };
		cell->cluster = cluster;
		cell->nextTimestep = nullptr;
		cell->numConnections = 0;
		if (j > 0 && j < cellsPerCluster - 1) {
			cell->numConnections = 2;
			cell->connections[0] = &cluster->cells[j - 1];
			cell->connections[1] = &cluster->cells[j + 1];
		}
		if (j == 0 && j < cellsPerCluster - 1) {
			cell->numConnections = 1;
			cell->connections[0] = &cluster->cells[j + 1];
		}
		if (j == cellsPerCluster - 1 && j > 0) {
			cell->numConnections = 1;
			cell->connections[0] = &cluster->cells[j - 1];
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
	int maxCellsPerCluster = 64;
	cudaData.clustersAC1 = ArrayController<ClusterCuda>(static_cast<int>(NUM_CLUSTERS * 1.1));
	cudaData.clustersAC2 = ArrayController<ClusterCuda>(static_cast<int>(NUM_CLUSTERS * 1.1));
	cudaData.cellsAC1 = ArrayController<CellCuda>(static_cast<int>(NUM_CLUSTERS * maxCellsPerCluster * 1.1));
	cudaData.cellsAC2 = ArrayController<CellCuda>(static_cast<int>(NUM_CLUSTERS * maxCellsPerCluster * 1.1));

	auto clusters = cudaData.clustersAC1.getArray(NUM_CLUSTERS);
	createCluster(&clusters[0], { 1500, 1200 }, { 0, 0 }, 90, 0, 64, size);
	createCluster(&clusters[1], { 1540, 1250 }, { -0.5, 0 }, 70, 0.0, 64, size);
	drawClusterToMap(&clusters[0], &cudaData);
	drawClusterToMap(&clusters[1], &cudaData);

/*
	for (int i = 0; i < NUM_CLUSTERS; ++i) {
		createCluster(&clusters[i], { random(size.x), random(size.y) }, { random(1.0f) - 0.5f, random(1.0) - 0.5f }, random(360.0f), random(0.1f) - 0.2f, 64, size);
		do {
			clusters[i].pos = { random(size.x), random(size.y) };
			centerCluster(&clusters[i]);
			updateAbsPos(&clusters[i]);

		} while (!isClusterPositionFree(&clusters[i], &cudaData));

		drawClusterToMap(&clusters[i], &cudaData);
		updateAngularMass(&clusters[i]);
	}
*/
}

void calcNextTimestep_Cuda()
{

	movement_Kernel <<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK, 0, cudaStream>>> (cudaData);
	cudaDeviceSynchronize();
	clearOldMap_Kernel <<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK, 0, cudaStream >>> (cudaData);
	cudaDeviceSynchronize();

	checkCudaErrors(cudaGetLastError());
	//****
	if (cudaData.clustersAC2.getEntireArray()[0].vel.x != -0.5 && cudaData.clustersAC2.getEntireArray()[1].vel.x != -0.5) {
		ClusterCuda temp1 = cudaData.clustersAC2.getEntireArray()[0];
		ClusterCuda temp2 = cudaData.clustersAC2.getEntireArray()[1];
		int dummy = 0;
	}
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

