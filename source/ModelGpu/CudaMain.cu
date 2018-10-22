#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <helper_cuda.h>
#include <list>

#include <stdio.h>
#include <functional>

#include "Base.cuh"
#include "Constants.cuh"
#include "CudaInterface.cuh"
#include "Simulation.cuh"
#include "SimulationDataManager.cuh"

//TODO: create class
namespace {
	int instances = 0;
	cudaStream_t cudaStream;
	SimulationDataManager *simulationManager;
}

void cudaInit(int2 const &size)
{
	if (instances++ == 0) {
		cudaStreamCreate(&cudaStream);
		cudaSetDevice(0);
		std::cout << "CUDA stream initialized" << std::endl;
	}

	simulationManager = new SimulationDataManager(size);
}


void cudaCalcNextTimestep()
{
	simulationManager->calcNextTimestep(cudaStream);
}

SimulationDataForAccess cudaGetData()
{
	return simulationManager->getDataForAccess();
}

void cudaSetData(SimulationDataForAccess const& access)
{
	simulationManager->setDataForAccess(access);
}

void cudaShutdown()
{
	cudaDeviceSynchronize();
	delete simulationManager;

	if (--instances == 0) {
		cudaDeviceReset();

		std::cout << "CUDA stream closed" << std::endl;
	}
}

