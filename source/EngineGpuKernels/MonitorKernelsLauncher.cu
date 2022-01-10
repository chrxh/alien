#include "MonitorKernelsLauncher.cuh"

#include "MonitorKernels.cuh"

void _MonitorKernelsLauncher::getMonitorData(GpuSettings const& gpuSettings, SimulationData const& data, CudaMonitorData const& monitorData)
{
    KERNEL_CALL_1_1(cudaGetCudaMonitorData, data, monitorData);
}
