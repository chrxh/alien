#include "TestKernelsLauncher.cuh"

#include "Macros.cuh"
#include "TestKernels.cuh"

void _TestKernelsLauncher::testOnly_mutate(GpuSettings const& gpuSettings, SimulationData const& data, uint64_t cellId, MutationType mutationType)
{
    KERNEL_CALL(cudaTestMutate, data, cellId, mutationType);
}
