#include "TestKernelsLauncher.cuh"

#include "Macros.cuh"
#include "TestKernels.cuh"

void _TestKernelsLauncher::testOnly_mutateCellFunction(GpuSettings const& gpuSettings, SimulationData const& data, uint64_t cellId)
{
    KERNEL_CALL(cudaMutateCellFunction, data, cellId);
}
