#include "TestKernelsLauncher.cuh"

#include "Macros.cuh"
#include "TestKernels.cuh"

void _TestKernelsLauncher::testOnly_mutateNeuron(GpuSettings const& gpuSettings, SimulationData const& data, uint64_t cellId)
{
}

void _TestKernelsLauncher::testOnly_mutateCellFunctionData(GpuSettings const& gpuSettings, SimulationData const& data, uint64_t cellId)
{
    KERNEL_CALL(cudaMutateCellFunctionData, data, cellId);
}

void _TestKernelsLauncher::testOnly_mutateCellFunction(GpuSettings const& gpuSettings, SimulationData const& data, uint64_t cellId)
{
}

void _TestKernelsLauncher::testOnly_mutateInsert(GpuSettings const& gpuSettings, SimulationData const& data, uint64_t cellId)
{
}

void _TestKernelsLauncher::testOnly_mutateDelete(GpuSettings const& gpuSettings, SimulationData const& data, uint64_t cellId)
{
}
