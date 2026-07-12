#include "StatisticsKernelsService.cuh"

#include <EngineKernels/StatisticsKernels.cuh>

void StatisticsKernelsService::init() {}

void StatisticsKernelsService::shutdown() {}

void StatisticsKernelsService::updateStatistics(CudaSettings const& gpuSettings, SimulationData const& data, SimulationStatistics const& simulationStatistics)
{
    KERNEL_CALL(cudaUpdateTimestepStatistics_substep1, data, simulationStatistics);
    KERNEL_CALL(cudaUpdateTimestepStatistics_substep2, data, simulationStatistics);
    KERNEL_CALL(cudaUpdateTimestepStatistics_substep3, data, simulationStatistics);
    KERNEL_CALL(cudaUpdateEvolutionStatistics_substep1, data, simulationStatistics);
    KERNEL_CALL(cudaUpdateEvolutionStatistics_substep2, data, simulationStatistics);
    KERNEL_CALL(cudaUpdateEvolutionStatistics_substep3, data, simulationStatistics);
    KERNEL_CALL(cudaUpdateEvolutionStatistics_substep4, data, simulationStatistics);
    KERNEL_CALL_1_1(cudaUpdateEvolutionStatistics_substep5, data, simulationStatistics);
    if (simulationStatistics.isLineageAccumulatorGCNeeded()) {
        KERNEL_CALL(cudaPrepareLineageAccumulatorGC, simulationStatistics);
        KERNEL_CALL(cudaLineageAccumulatorGC, simulationStatistics);
        KERNEL_CALL_1_1(cudaFinishLineageAccumulatorGC, simulationStatistics);
    }
    KERNEL_CALL_1_1(cudaUpdateHistogramData_substep1, data, simulationStatistics);
    KERNEL_CALL(cudaUpdateHistogramData_substep2, data, simulationStatistics);
    KERNEL_CALL(cudaUpdateHistogramData_substep3, data, simulationStatistics);
}
