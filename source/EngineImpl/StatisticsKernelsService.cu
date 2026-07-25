#include "StatisticsKernelsService.cuh"

#include <EngineKernels/StatisticsKernels.cuh>

void StatisticsKernelsService::init() {}

void StatisticsKernelsService::shutdown() {}

void StatisticsKernelsService::updateStatistics(
    CudaSettings const& gpuSettings,
    SimulationData const& data,
    SimulationStatistics const& simulationStatistics)
{
    KERNEL_CALL(cudaResetStatistics, data, simulationStatistics);
    KERNEL_CALL(cudaCollectObjectAndCreatureStatistics, data, simulationStatistics);
    KERNEL_CALL(cudaCollectGenomeAndEnergyStatistics, data, simulationStatistics);
    KERNEL_CALL(cudaCompactLineageStatistics, simulationStatistics);
    if (simulationStatistics.isLineageAccumulatorGCNeeded()) {
        KERNEL_CALL(cudaPrepareLineageAccumulatorGC, simulationStatistics);
        KERNEL_CALL(cudaLineageAccumulatorGC, simulationStatistics);
        KERNEL_CALL_1_1(cudaFinishLineageAccumulatorGC, simulationStatistics);
    }
}
