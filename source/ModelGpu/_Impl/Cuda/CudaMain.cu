#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <helper_cuda.h>

#include <stdio.h>
#include <functional>

#include "Base.cuh"
#include "Constants.cuh"
#include "CudaInterface.cuh"
#include "Simulation.cuh"
#include "HostHelper.cuh"

namespace {
	cudaStream_t cudaStream;
	CudaSimulationManager *cudaSimulationManager;
}

void createCluster(CellClusterData* cluster, float2 pos, float2 vel, float angle, float angVel, float energy, int2 clusterSize, int2 const &size)
{
	cluster->pos = pos;
	cluster->vel = vel;
	cluster->angle = angle;
	cluster->angularVel = angVel;
	cluster->numCells = clusterSize.x * clusterSize.y;
	cluster->cells = cudaSimulationManager->data.cellsAC1.getArray(clusterSize.x*clusterSize.y);

	for (int x = 0; x < clusterSize.x; ++x) {
		for (int y = 0; y < clusterSize.y; ++y) {
			CellData *cell = &cluster->cells[x + y*clusterSize.x];
			cell->relPos = { static_cast<float>(x), static_cast<float>(y) };
			cell->cluster = cluster;
			cell->nextTimestep = nullptr;
			cell->protectionCounter = 0;
			cell->numConnections = 0;
			cell->energy = energy;
			cell->setProtectionCounterForNextTimestep = false;
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

	cudaSimulationManager = new CudaSimulationManager(size);
	
	auto clusters = cudaSimulationManager->data.clustersAC1.getArray(NUM_CLUSTERS);

	for (int i = 0; i < NUM_CLUSTERS; ++i) {
		createCluster(&clusters[i], { 0.0f, 0.0f }, { random(1.0f) - 0.5f, random(1.0f) - 0.5f }, random(360.0f), random(0.4f) - 0.2f, 100.0, { rand() % 20 + 1, rand() % 20 + 1 }, size);
		do {
			clusters[i].pos = { random(static_cast<float>(size.x)), random(static_cast<float>(size.y)) };
			centerCluster(&clusters[i]);
			updateAbsPos(&clusters[i]);

		} while (!isClusterPositionFree(&clusters[i], &cudaSimulationManager->data));

		drawClusterToMap(&clusters[i], &cudaSimulationManager->data);
		updateAngularMass(&clusters[i]);
	}

}


void cudaCalcNextTimestep()
{
	cudaSimulationManager->prepareTargetData();

	clusterMovement <<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK, 0, cudaStream>>> (cudaSimulationManager->data);
	cudaDeviceSynchronize();
	particleMovement << <NUM_BLOCKS, NUM_THREADS_PER_BLOCK, 0, cudaStream >> > (cudaSimulationManager->data);
	cudaDeviceSynchronize();
	clearMaps <<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK, 0, cudaStream>>> (cudaSimulationManager->data);
	cudaDeviceSynchronize();

	checkCudaErrors(cudaGetLastError());
	
	cudaSimulationManager->swapData();
}

DataForAccess cudaGetData()
{
	return cudaSimulationManager->getDataForAccess();
}


void cudaShutdown()
{
	cudaDeviceSynchronize();
	cudaDeviceReset();

	delete cudaSimulationManager;
}

