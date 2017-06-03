#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <helper_cuda.h>

#include <stdio.h>
#include <functional>

#include "CudaShared.cuh"
#include "CudaBase.cuh"
#include "CudaDeviceFunctions.cuh"

#define NUM_THREADS_PER_BLOCK 32
#define NUM_BLOCKS (32 * 5) /*160*/
#define NUM_CLUSTERS (NUM_BLOCKS * 50)

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
	cudaData.numClusters1 = NUM_CLUSTERS;
	cudaData.clustersAC1 = ArrayController<ClusterCuda>(NUM_CLUSTERS * 2);
	cudaData.cellsAC1 = ArrayController<CellCuda>(NUM_CLUSTERS * cellsPerCluster * 2);
	cudaData.clustersAC2 = ArrayController<ClusterCuda>(NUM_CLUSTERS * 2);
	cudaData.cellsAC2 = ArrayController<CellCuda>(NUM_CLUSTERS * cellsPerCluster * 2);

	cudaData.clusters1 = cudaData.clustersAC1.getArray(NUM_CLUSTERS);
	for (int i = 0; i < NUM_CLUSTERS; ++i) {
		cudaData.clusters1[i].pos = { random(size.x), random(size.y) };
		cudaData.clusters1[i].vel = { random(1.0f) - 0.5f, random(1.0) - 0.5f };
		cudaData.clusters1[i].angle = random(360.0f);
		cudaData.clusters1[i].angularVel = random(10.0f) - 5.0f;
		cudaData.clusters1[i].numCells = cellsPerCluster;

		cudaData.clusters1[i].cells = cudaData.cellsAC1.getArray(cellsPerCluster);
		for (int j = 0; j < cellsPerCluster; ++j) {
			cudaData.clusters1[i].cells[j].relPos = { j - 20.0f, j - 20.0f };
		}

	}
}

void calcNextTimestep_Cuda()
{
	movement_Kernel <<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK, 0, cudaStream>>> (cudaData);
	checkCudaErrors(cudaGetLastError());
	cudaDeviceSynchronize();
}

void getDataRef_Cuda(int& numClusters, ClusterCuda*& clusters)
{
	numClusters = cudaData.numClusters1;
	clusters = cudaData.clusters1;
}


void end_Cuda()
{
	cudaDeviceSynchronize();
	cudaDeviceReset();

	cudaData.cellsAC2.free();
	cudaData.clustersAC2.free();
	cudaData.cellsAC1.free();
	cudaData.clustersAC1.free();
	cudaFree(cudaData.map1);
	cudaFree(cudaData.map1);
}

/*
int main()
{
	cudaSetDevice(0);
	cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1024 * 1024 * 124);

	Config config{ size };

	CellMap map;
	size_t mapSize = size.x * size.y * sizeof(int) * layers;
	cudaMallocManaged(&map.map1, mapSize);
	cudaMallocManaged(&map.map2, mapSize);
	for (int i = 0; i < size.x * size.y * layers; ++i) {
		map.map1[i] = -1;
		map.map1[i] = -1;
		map.map2[i] = -1;
		map.map2[i] = -1;
	}
	int cellsPerCluster = 32;
	ArrayController<Cluster> clustersAC(numClusters * 2);
	ArrayController<Cell> cellsAC(numClusters * cellsPerCluster * 2);
	ArrayController<CellMapEntry> cellMapInfoAC1(numClusters * cellsPerCluster * 2);
	ArrayController<CellMapEntry> cellMapInfoAC2(numClusters * cellsPerCluster * 2);

	Cluster* clusters = clustersAC.getArray(numClusters);
	for (int i = 0; i < numClusters; ++i) {
		clusters[i].pos = { random(size.x), random(size.y) };
		clusters[i].vel = { random(1.0f) - 0.5f, random(1.0) - 0.5f };
		clusters[i].angle = random(360.0f);
		clusters[i].angularVel = random(10.0f) - 5.0f;
		clusters[i].numCells = cellsPerCluster;

		clusters[i].cells = cellsAC.getArray(cellsPerCluster);
		for (int j = 0; j < cellsPerCluster; ++j) {
			clusters[i].cells[j].relPos = { j - 20.0f, j - 20.0f };
		}

	}

	printf("%f, %f\n", clusters[320].pos.x, clusters[320].vel.x);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);

	for (int p = 0; p < 1000; ++p) {
		movement_Kernel <<<numBlocks, numThreadsPerBlock >>> (clusters, map.map1, map.map2, cellMapInfoAC1, cellMapInfoAC2, config);
		evaluate();
		cudaDeviceSynchronize();
	}


	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("Elapsed: %f\n", elapsedTime);

	printf(" = %f\n", clusters[320].pos.x);

	cudaDeviceReset();

	cellMapInfoAC2.free();
	cellMapInfoAC1.free();
	cellsAC.free();
	clustersAC.free();
	cudaFree(map.map1);
	cudaFree(map.map1);
	return 0;
}
*/
