#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <helper_cuda.h>

#include <stdio.h>
#include <functional>

#include "CudaBase.cuh"
#include "CudaConstants.cuh"
#include "CudaShared.cuh"
#include "CudaDeviceFunctions.cuh"

cudaStream_t cudaStream;
CudaData cudaData;

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
		cudaData.map1[i] = nullptr;
		cudaData.map2[i] = nullptr;
		cudaData.map2[i] = nullptr;
	}
	int cellsPerCluster = 32;
	cudaData.clustersAC1 = ArrayController<ClusterCuda>(NUM_CLUSTERS * 2);
	cudaData.cellsAC1 = ArrayController<CellCuda>(NUM_CLUSTERS * cellsPerCluster * 2);
	cudaData.clustersAC2 = ArrayController<ClusterCuda>(NUM_CLUSTERS * 2);
	cudaData.cellsAC2 = ArrayController<CellCuda>(NUM_CLUSTERS * cellsPerCluster * 2);

	auto clusters = cudaData.clustersAC1.getArray(NUM_CLUSTERS);
	for (int i = 0; i < NUM_CLUSTERS; ++i) {
		clusters[i].pos = { random(size.x), random(size.y) };
		clusters[i].vel = { random(1.0f) - 0.5f, random(1.0) - 0.5f };
		clusters[i].angle = random(360.0f);
		clusters[i].angularVel = random(10.0f) - 5.0f;
		clusters[i].numCells = cellsPerCluster;

		clusters[i].cells = cudaData.cellsAC1.getArray(cellsPerCluster);
		for (int j = 0; j < cellsPerCluster; ++j) {
			CellCuda *cell = &clusters[i].cells[j];
			cell->relPos = { j - 20.0f, j - 20.0f };
			cell->absPos = clusters[i].pos;
			cell->cluster = &clusters[i];
			cell->nextTimestep = nullptr;
			if (j > 0 && j < cellsPerCluster - 1) {
				cell->numConnections = 2;
				cell->connections[0] = &clusters[i].cells[j - 1];
				cell->connections[1] = &clusters[i].cells[j + 1];
			}
			if (j == 0) {
				cell->numConnections = 1;
				cell->connections[0] = &clusters[i].cells[j + 1];
			}
			if (j == cellsPerCluster - 1) {
				cell->numConnections = 1;
				cell->connections[0] = &clusters[i].cells[j - 1];
			}
		}

	}
}

void calcNextTimestep_Cuda()
{
	movement_Kernel <<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK, 0, cudaStream>>> (cudaData);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	swap(cudaData.clustersAC1, cudaData.clustersAC2);
	swap(cudaData.cellsAC1, cudaData.cellsAC2);
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

