#include "StatisticsKernelsService.cuh"

#include <EngineKernels/KernelLauncher.cuh>
#include <EngineKernels/StatisticsKernels.cuh>

void StatisticsKernelsService::init() {}

void StatisticsKernelsService::shutdown() {}

void StatisticsKernelsService::updateStatistics(
    KernelLaunchSettings const& gpuSettings,
    SimulationData const& data,
    SimulationStatistics const& simulationStatistics)
{
    launchKernelOnDefaultStream(KERNEL(cudaResetStatistics), LaunchConfig{gpuSettings.numBlocks, 8}, data, simulationStatistics);
    launchKernelOnDefaultStream(KERNEL(cudaCollectObjectAndCreatureStatistics), LaunchConfig{gpuSettings.numBlocks, 8}, data, simulationStatistics);
    launchKernelOnDefaultStream(KERNEL(cudaCollectGenomeAndEnergyStatistics), LaunchConfig{gpuSettings.numBlocks, 8}, data, simulationStatistics);
    launchKernelOnDefaultStream(KERNEL(cudaCompactLineageStatistics), LaunchConfig{gpuSettings.numBlocks, 8}, simulationStatistics);
    if (simulationStatistics.isLineageAccumulatorGCNeeded()) {
        launchKernelOnDefaultStream(KERNEL(cudaPrepareLineageAccumulatorGC), LaunchConfig{gpuSettings.numBlocks, 8}, simulationStatistics);
        launchKernelOnDefaultStream(KERNEL(cudaLineageAccumulatorGC), LaunchConfig{gpuSettings.numBlocks, 8}, simulationStatistics);
        launchKernelOnDefaultStream(KERNEL(cudaFinishLineageAccumulatorGC), LaunchConfig{1, 1}, simulationStatistics);
    }
}
