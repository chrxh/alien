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

	std::cout << "CUDA stream initialized" << std::endl;

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
	cudaDeviceReset();

	std::cout << "CUDA stream closed" << std::endl;
}

