#include "MonitorKernels.cuh"

namespace
{
/*
    __global__ void getEnergyForMonitorData(SimulationData data, CudaMonitorData monitorData)
    {
        {
            auto& cells = data.entities.cellPointers;
            auto const partition = calcPartition(cells.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

            for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
                auto& cell = cells.at(index);
                monitorData.incInternalEnergy(cell->energy);
            }
        }
        {
            auto& particles = data.entities.particlePointers;
            auto const partition = calcPartition(particles.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

            for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
                auto& particle = particles.at(index);
                monitorData.incInternalEnergy(particle->energy);
            }
        }
        {
            auto& tokens = data.entities.tokenPointers;
            auto const partition = calcPartition(tokens.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

            for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
                auto& token = tokens.at(index);
                monitorData.incInternalEnergy(token->energy);
            }
        }
    }
*/
}

__global__ void cudaGetCudaMonitorData_substep1(SimulationData data, CudaMonitorData monitorData)
{
    monitorData.reset();

    monitorData.setNumParticles(data.objects.particlePointers.getNumEntries());

    //    KERNEL_CALL(getEnergyForMonitorData, data, monitorData);
}

__global__ void cudaGetCudaMonitorData_substep2(SimulationData data, CudaMonitorData monitorData)
{
    auto& cells = data.objects.cellPointers;
    auto const partition = calcPartition(cells.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cells.at(index);
        monitorData.incNumCell(calcMod(cell->color, 7));
        monitorData.incNumConnections(cell->numConnections);
    }
}

__global__ void cudaGetCudaMonitorData_substep3(SimulationData data, CudaMonitorData monitorData)
{
    monitorData.halveNumConnections();
}
