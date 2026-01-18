#include "StatisticsKernels.cuh"

__global__ void cudaUpdateTimestepStatistics_substep1(SimulationData data, SimulationStatistics statistics)
{
    statistics.resetTimestepData();
}

__global__ void cudaUpdateTimestepStatistics_substep2(SimulationData data, SimulationStatistics statistics)
{
    {
        auto& objects = data.entities.objects;
        auto const partition = calcSystemThreadPartition(objects.getNumEntries());

        for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
            auto& object = objects.at(index);
            statistics.incNumCells(object->color);
            if (object->type == ObjectType_FreeCell) {
                statistics.incNumFreeCells(object->color);
                statistics.addEnergy(object->color, object->typeData.freeCell.rawEnergy);
            } else if (object->type == ObjectType_Cell) {
                statistics.addEnergy(object->color, object->typeData.cell.getEnergy());
            }
            //if (object->typeData.cell.cellType == CellType_Constructor && GenomeDecoder::containsSelfReplication(object->typeData.cell.cellTypeData.constructor)) {
            //    statistics.incNumReplicator(object->color);
            //    statistics.incMutant(object->color, object->lineageId, object->numObjects);
            //    auto numNodes = GenomeDecoder::getNumNodesRecursively(object->typeData.cell.cellTypeData.constructor.genome, object->typeData.cell.cellTypeData.constructor.genomeSize, true, true);
            //    statistics.addNumGenomeNodes(object->color, numNodes);
            //    statistics.addNumCells(object->color, object->numObjects);
            //}
            //if (object->typeData.cell.cellType == CellType_Injector && GenomeDecoder::containsSelfReplication(object->typeData.cell.cellTypeData.injector)) {
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
        //auto& objects = data.entities.objects;
        //auto const partition = calcAllThreadsPartition(cells.getNumEntries());

        //auto numReplicators = toDouble(statistics.getNumReplicators());
        //auto averageNumCells = statistics.getSummedNumCells() / numReplicators;

        //for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        //    auto& object = cells.at(index);
        //if (object->typeData.cell.cellType == CellType_Constructor && GenomeDecoder::containsSelfReplication(object->typeData.cell.cellTypeData.constructor)) {
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
    auto& objects = data.entities.objects;
    auto const partition = calcSystemThreadPartition(objects.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = objects.at(index);
        if (object->fixed) {
            continue;
        }
        if (object->type == ObjectType_Cell) {
            statistics.maxValue(object->typeData.cell.age);
        } else if (object->type == ObjectType_FreeCell) {
            statistics.maxValue(object->typeData.freeCell.age);
        }
    }
}

__global__ void cudaUpdateHistogramData_substep3(SimulationData data, SimulationStatistics statistics)
{
    auto& objects = data.entities.objects;
    auto const partition = calcSystemThreadPartition(objects.getNumEntries());

    auto maxAge = statistics.getMaxValue();
    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = objects.at(index);
        if (object->fixed) {
            continue;
        }
        if (object->type == ObjectType_Cell) {
            auto slot = object->typeData.cell.age * MAX_HISTOGRAM_SLOTS / (maxAge + 1);
            statistics.incNumCells(object->color, slot);
        } else if (object->type == ObjectType_FreeCell) {
            auto slot = object->typeData.freeCell.age * MAX_HISTOGRAM_SLOTS / (maxAge + 1);
            statistics.incNumCells(object->color, slot);
        }
    }
}
