#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <helper_cuda.h>

#include <stdio.h>
#include <functional>

#include "Base.cuh"
#include "Constants.cuh"
#include "CudaInterface.cuh"
#include "Simulation.cuh"
#include "SimulationDataManager.cuh"

namespace {
	cudaStream_t cudaStream;
	SimulationDataManager *simulationManager;
}

void cudaInit(int2 const &size)
{
	cudaStreamCreate(&cudaStream);
	cudaSetDevice(0);

	simulationManager = new SimulationDataManager(size);
	
	auto clusters = simulationManager->data.clustersAC1.getArray(NUM_CLUSTERS);

	for (int i = 0; i < NUM_CLUSTERS; ++i) {
		simulationManager->createCluster(simulationManager->data, &clusters[i], { 0.0f, 0.0f }, { random(1.0f) - 0.5f, random(1.0f) - 0.5f }, random(360.0f), random(0.4f) - 0.2f, 100.0, { rand() % 20 + 1, rand() % 20 + 1 }, size);
		do {
			clusters[i].pos = { random(static_cast<float>(size.x)), random(static_cast<float>(size.y)) };
			simulationManager->centerCluster(&clusters[i]);
			simulationManager->updateAbsPos(&clusters[i]);

		} while (!simulationManager->isClusterPositionFree(&clusters[i], &simulationManager->data));

		simulationManager->drawClusterToMap(&clusters[i], &simulationManager->data);
		simulationManager->updateAngularMass(&clusters[i]);
	}
}


void cudaCalcNextTimestep()
{
	simulationManager->prepareTargetData();

	clusterMovement <<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK, 0, cudaStream>>> (simulationManager->data);
	cudaDeviceSynchronize();
	particleMovement << <NUM_BLOCKS, NUM_THREADS_PER_BLOCK, 0, cudaStream >> > (simulationManager->data);
	cudaDeviceSynchronize();
	clearMaps <<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK, 0, cudaStream>>> (simulationManager->data);
	cudaDeviceSynchronize();

	checkCudaErrors(cudaGetLastError());
	
	simulationManager->swapData();
}

SimulationDataForAccess cudaGetData()
{
	return simulationManager->getDataForAccess();
}

void cudaShutdown()
{
	cudaDeviceSynchronize();
	delete simulationManager;
	cudaDeviceReset();

}

