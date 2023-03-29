#include "StatisticsKernels.cuh"

namespace
{
/*
    __global__ void getEnergyForMonitorData(SimulationData data, SimulationStatistics monitorData)
    {
        {
            auto& cells = data.entities.cellPointers;
            auto const partition = calcAllThreadsPartition(cells.getNumEntries());

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

__global__ void cudaUpdateTimestepStatistics_substep1(SimulationData data, SimulationStatistics statistics)
{
    statistics.resetTimestepData();

    statistics.setNumParticles(data.objects.particlePointers.getNumEntries());

    //    KERNEL_CALL(getEnergyForMonitorData, data, statistics);
}

__global__ void cudaUpdateTimestepStatistics_substep2(SimulationData data, SimulationStatistics statistics)
{
    auto& cells = data.objects.cellPointers;
    auto const partition = calcAllThreadsPartition(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cells.at(index);
        statistics.incNumCell(cell->color);
        statistics.incNumConnections(cell->numConnections);
    }
}

__global__ void cudaUpdateTimestepStatistics_substep3(SimulationData data, SimulationStatistics statistics)
{
    statistics.halveNumConnections();
}

__global__ void cudaUpdateHistogramData_substep1(SimulationData data, SimulationStatistics statistics)
{
    statistics.resetHistogramData();
}

__global__ void cudaUpdateHistogramData_substep2(SimulationData data, SimulationStatistics statistics)
{
    auto& cells = data.objects.cellPointers;
    auto const partition = calcAllThreadsPartition(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cells.at(index);
        if (cell->barrier) {
            continue;
        }
        statistics.maxValue(cell->age);
    }
}

__global__ void cudaUpdateHistogramData_substep3(SimulationData data, SimulationStatistics statistics)
{
    auto& cells = data.objects.cellPointers;
    auto const partition = calcAllThreadsPartition(cells.getNumEntries());

    auto maxAge = statistics.getMaxValue();
    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cells.at(index);
        if (cell->barrier) {
            continue;
        }
        auto slot = cell->age * MAX_HISTOGRAM_SLOTS / (maxAge + 1);
        statistics.incNumCell(cell->color, slot);
    }
}
