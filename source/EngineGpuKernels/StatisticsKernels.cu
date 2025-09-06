#include "StatisticsKernels.cuh"

__global__ void cudaUpdateTimestepStatistics_substep1(SimulationData data, SimulationStatistics statistics)
{
    statistics.resetTimestepData();
}

__global__ void cudaUpdateTimestepStatistics_substep2(SimulationData data, SimulationStatistics statistics)
{
    {
        auto& cells = data.objects.cells;
        auto const partition = calcAllThreadsPartition(cells.getNumEntries());

        for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
            auto& cell = cells.at(index);
            statistics.incNumCells(cell->color);
            if (cell->cellType == CellType_Free) {
                statistics.incNumFreeCells(cell->color);
            }
            statistics.addEnergy(cell->color, cell->energy);
            //if (cell->cellType == CellType_Constructor && GenomeDecoder::containsSelfReplication(cell->cellTypeData.constructor)) {
            //    statistics.incNumReplicator(cell->color);
            //    statistics.incMutant(cell->color, cell->lineageId, cell->numCells);
            //    auto numNodes = GenomeDecoder::getNumNodesRecursively(cell->cellTypeData.constructor.genome, cell->cellTypeData.constructor.genomeSize, true, true);
            //    statistics.addNumGenomeNodes(cell->color, numNodes);
            //    statistics.addNumCells(cell->color, cell->numCells);
            //}
            //if (cell->cellType == CellType_Injector && GenomeDecoder::containsSelfReplication(cell->cellTypeData.injector)) {
            //    statistics.incNumViruses(cell->color);
            //}
        }
    }
    {
        auto& particles = data.objects.particles;
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
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        statistics.halveNumConnections();
    }

    {
        //auto& cells = data.objects.cells;
        //auto const partition = calcAllThreadsPartition(cells.getNumEntries());

        //auto numReplicators = toDouble(statistics.getNumReplicators());
        //auto averageNumCells = statistics.getSummedNumCells() / numReplicators;

        //for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        //    auto& cell = cells.at(index);
            //if (cell->cellType == CellType_Constructor && GenomeDecoder::containsSelfReplication(cell->cellTypeData.constructor)) {
            //    auto variance = toDouble(cell->numCells) - averageNumCells;
            //    variance = variance * variance / numReplicators;
            //    statistics.addToNumCellsVariance(cell->color, variance);
            //}
        //}
    }
}

__global__ void cudaUpdateHistogramData_substep1(SimulationData data, SimulationStatistics statistics)
{
    statistics.resetHistogramData();
}

__global__ void cudaUpdateHistogramData_substep2(SimulationData data, SimulationStatistics statistics)
{
    auto& cells = data.objects.cells;
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
    auto& cells = data.objects.cells;
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
