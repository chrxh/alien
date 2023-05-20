#include "StatisticsKernels.cuh"

#include "GenomeDecoder.cuh"

__global__ void cudaUpdateTimestepStatistics_substep1(SimulationData data, SimulationStatistics statistics)
{
    statistics.resetTimestepData();
}

__global__ void cudaUpdateTimestepStatistics_substep2(SimulationData data, SimulationStatistics statistics)
{
    {
        auto& cells = data.objects.cellPointers;
        auto const partition = calcAllThreadsPartition(cells.getNumEntries());

        for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
            auto& cell = cells.at(index);
            statistics.incNumCells(cell->color);
            statistics.incNumConnections(cell->color, cell->numConnections);
            statistics.addEnergy(cell->color, cell->energy);
            if (cell->cellFunction == CellFunction_Constructor && GenomeDecoder::containsSelfReplication(cell->cellFunctionData.constructor)) {
                statistics.incNumReplicator(cell->color);
            }
            if (cell->cellFunction == CellFunction_Injector && cell->cellFunctionData.injector.mode == InjectorMode_InjectAll) {
                statistics.incNumReplicator(cell->color);
            }
            if (cell->cellFunction == CellFunction_Constructor) {
                statistics.addNumGenomeBytes(cell->color, cell->cellFunctionData.constructor.genomeSize);
                statistics.incNumGenomes(cell->color);
            }
            if (cell->cellFunction == CellFunction_Injector) {
                statistics.addNumGenomeBytes(cell->color, cell->cellFunctionData.injector.genomeSize);
                statistics.incNumGenomes(cell->color);
            }
        }
    }
    {
        auto& particles = data.objects.particlePointers;
        auto const partition = calcAllThreadsPartition(particles.getNumEntries());

        for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
            auto& particle = particles.at(index);
            statistics.incNumParticles(particle->color);
            statistics.addEnergy(particle->color, particle->energy);
        }
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
        statistics.incNumCells(cell->color, slot);
    }
}
