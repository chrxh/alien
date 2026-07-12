#include "ConstantMemory.cuh"

__constant__ GpuSettings cudaThreadSettings;
__device__ char cudaSimulationParametersData[sizeof(SimulationParameters)];
