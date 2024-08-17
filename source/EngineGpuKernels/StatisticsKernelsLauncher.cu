#include "StatisticsKernelsLauncher.cuh"

#include "StatisticsKernels.cuh"

void _StatisticsKernelsLauncher::updateStatistics(GpuSettings const& gpuSettings, SimulationData const& data, SimulationStatistics const& simulationStatistics)
{
    KERNEL_CALL(cudaUpdateTimestepStatistics_substep1, data, simulationStatistics);
    KERNEL_CALL(cudaUpdateTimestepStatistics_substep2, data, simulationStatistics);
    KERNEL_CALL(cudaUpdateTimestepStatistics_substep3, data, simulationStatistics);
    KERNEL_CALL_1_1(cudaUpdateHistogramData_substep1, data, simulationStatistics);
    KERNEL_CALL(cudaUpdateHistogramData_substep2, data, simulationStatistics);
    KERNEL_CALL(cudaUpdateHistogramData_substep3, data, simulationStatistics);
}
