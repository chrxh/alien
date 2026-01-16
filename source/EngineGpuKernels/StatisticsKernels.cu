#include "StatisticsKernels.cuh"

__global__ void cudaUpdateTimestepStatistics_substep1(SimulationData data, SimulationStatistics statistics)
{
    statistics.resetTimestepData();
}

__global__ void cudaUpdateTimestepStatistics_substep2(SimulationData data, SimulationStatistics statistics)
{
    {
        auto& cells = data.entities.objects;
        auto const partition = calcSystemThreadPartition(cells.getNumEntries());

        for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
            auto& object = cells.at(index);
            statistics.incNumCells(object->color);
            if (object->cellType == CellType_Free) {
                statistics.incNumFreeCells(object->color);
            }
            statistics.addEnergy(object->color, object->getEnergy());
            //if (object->cellType == CellType_Constructor && GenomeDecoder::containsSelfReplication(object->cellTypeData.constructor)) {
            //    statistics.incNumReplicator(object->color);
            //    statistics.incMutant(object->color, object->lineageId, object->numObjects);
            //    auto numNodes = GenomeDecoder::getNumNodesRecursively(object->cellTypeData.constructor.genome, object->cellTypeData.constructor.genomeSize, true, true);
            //    statistics.addNumGenomeNodes(object->color, numNodes);
            //    statistics.addNumCells(object->color, object->numObjects);
            //}
            //if (object->cellType == CellType_Injector && GenomeDecoder::containsSelfReplication(object->cellTypeData.injector)) {
            //    statistics.incNumViruses(object->color);
            //}
        }
    }
    {
        auto& particles = data.entities.energies;
        auto const partition = calcSystemThreadPartition(particles.getNumEntries());

        for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
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
        //auto& cells = data.entities.objects;
        //auto const partition = calcAllThreadsPartition(cells.getNumEntries());

        //auto numReplicators = toDouble(statistics.getNumReplicators());
        //auto averageNumCells = statistics.getSummedNumCells() / numReplicators;

        //for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        //    auto& object = cells.at(index);
        //if (object->cellType == CellType_Constructor && GenomeDecoder::containsSelfReplication(object->cellTypeData.constructor)) {
        //    auto variance = toDouble(object->numObjects) - averageNumCells;
        //    variance = variance * variance / numReplicators;
        //    statistics.addToNumCellsVariance(object->color, variance);
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
    auto& cells = data.entities.objects;
    auto const partition = calcSystemThreadPartition(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = cells.at(index);
        if (object->fixed) {
            continue;
        }
        statistics.maxValue(object->age);
    }
}

__global__ void cudaUpdateHistogramData_substep3(SimulationData data, SimulationStatistics statistics)
{
    auto& cells = data.entities.objects;
    auto const partition = calcSystemThreadPartition(cells.getNumEntries());

    auto maxAge = statistics.getMaxValue();
    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = cells.at(index);
        if (object->fixed) {
            continue;
        }
        auto slot = object->age * MAX_HISTOGRAM_SLOTS / (maxAge + 1);
        statistics.incNumCells(object->color, slot);
    }
}
