#include "TestKernelsLauncher.cuh"

#include "Macros.cuh"
#include "TestKernels.cuh"

void _TestKernelsLauncher::testOnly_mutate(GpuSettings const& gpuSettings, SimulationData const& data, uint64_t cellId, MutationType mutationType)
{
    KERNEL_CALL(cudaTestMutate, data, cellId, mutationType);
}

void _TestKernelsLauncher::testOnly_mutationCheck(GpuSettings const& gpuSettings, SimulationData const& data, uint64_t cellId)
{
    KERNEL_CALL(cudaTestMutationCheck, data, cellId);
}

void _TestKernelsLauncher::testOnly_createConnection(GpuSettings const& gpuSettings, SimulationData const& data, uint64_t cellId1, uint64_t cellId2)
{
    KERNEL_CALL_1_1(cudaTestCreateConnection, data, cellId1, cellId2);
}
