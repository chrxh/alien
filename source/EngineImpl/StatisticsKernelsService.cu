#include "StatisticsKernelsService.cuh"

#include <EngineKernels/KernelLauncher.cuh>
#include <EngineKernels/StatisticsKernels.cuh>

void StatisticsKernelsService::init() {}

void StatisticsKernelsService::shutdown() {}

void StatisticsKernelsService::updateStatistics(
    KernelLaunchSettings const& launchSettings,
    SimulationData const& data,
    SimulationStatistics const& simulationStatistics)
{
    launchKernelOnDefaultStream(KERNEL(cudaResetStatistics), LaunchConfig{launchSettings.numBlocks, 8}, data, simulationStatistics);
    launchKernelOnDefaultStream(KERNEL(cudaCollectObjectAndCreatureStatistics), LaunchConfig{launchSettings.numBlocks, 8}, data, simulationStatistics);
    launchKernelOnDefaultStream(KERNEL(cudaCollectGenomeAndEnergyStatistics), LaunchConfig{launchSettings.numBlocks, 8}, data, simulationStatistics);
    launchKernelOnDefaultStream(KERNEL(cudaCompactLineageStatistics), LaunchConfig{launchSettings.numBlocks, 8}, simulationStatistics);
    if (simulationStatistics.isLineageAccumulatorGCNeeded()) {
        launchKernelOnDefaultStream(KERNEL(cudaPrepareLineageAccumulatorGC), LaunchConfig{launchSettings.numBlocks, 8}, simulationStatistics);
        launchKernelOnDefaultStream(KERNEL(cudaLineageAccumulatorGC), LaunchConfig{launchSettings.numBlocks, 8}, simulationStatistics);
        launchKernelOnDefaultStream(KERNEL(cudaFinishLineageAccumulatorGC), LaunchConfig{1, 1}, simulationStatistics);
    }
}
