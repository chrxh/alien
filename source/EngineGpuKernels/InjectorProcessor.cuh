#pragma once

#include "SignalProcessor.cuh"
#include "SimulationData.cuh"

class InjectorProcessor
{
public:
    __inline__ __device__ static void process(SimulationData& data, SimulationStatistics& statistics);

private:
    __inline__ __device__ static void processCell(SimulationData& data, SimulationStatistics& statistics, Cell* cell);
    __inline__ __device__ static int countAndTrackDefenderCells(SimulationStatistics& statistics, Cell* cell);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__device__ __inline__ void InjectorProcessor::process(SimulationData& data, SimulationStatistics& statistics)
{
    auto& operations = data.cellTypeOperations[CellType_Injector];
    auto partition = calcAllThreadsPartition(operations.getNumEntries());
    for (int i = partition.startIndex; i <= partition.endIndex; ++i) {
        processCell(data, statistics, operations.at(i).cell);
    }
}

__inline__ __device__ void InjectorProcessor::processCell(SimulationData& data, SimulationStatistics& statistics, Cell* cell)
{
    if (abs(cell->signal.channels[0]) >= TRIGGER_THRESHOLD) {

        auto& injector = cell->cellTypeData.injector;

        bool match = false;
        bool injection = false;

        switch (injector.mode) {
        case InjectorMode_InjectAll: {
            data.cellMap.executeForEach(
                cell->pos, cudaSimulationParameters.injectorInjectionRadius.value[cell->color], cell->detached, [&](Cell* const& otherCell) {
                    if (cell == otherCell) {
                        return;
                    }
                    if (otherCell->cellType != CellType_Constructor && otherCell->cellType != CellType_Injector) {
                        return;
                    }
                    //if (otherCell->livingState == LivingState_UnderConstruction) {
                    //    return;
                    //}
                    if (otherCell->cellTypeData.constructor.genomeCurrentNodeIndex != 0) {
                        return;
                    }
                    if (!otherCell->tryLock()) {
                        return;
                    }
                    match = true;
                    auto injectorDuration = cudaSimulationParameters.injectorInjectionTime.value[cell->color][otherCell->color];

                    auto numDefenderCells = countAndTrackDefenderCells(statistics, otherCell);
                    float defendStrength = numDefenderCells == 0
                        ? 1.0f : powf(cudaSimulationParameters.defenderAntiInjectorStrength.value[cell->color] + 1.0f, numDefenderCells);
                    injectorDuration = toInt(toFloat(injectorDuration) * defendStrength);
                    if (injector.counter < injectorDuration) {
                        otherCell->releaseLock();
                        return;
                    }

                    auto targetGenome = data.objects.heap.getRawSubArray(injector.genomeSize);
                    for (int i = 0; i < injector.genomeSize; ++i) {
                        targetGenome[i] = injector.genome[i];
                    }

                    if (otherCell->cellType == CellType_Constructor) {
                        otherCell->cellTypeData.constructor.genome = targetGenome;
                        otherCell->cellTypeData.constructor.genomeSize = injector.genomeSize;
                        otherCell->cellTypeData.constructor.numInheritedGenomeNodes = 0;
                    } else {
                        otherCell->cellTypeData.injector.genome = targetGenome;
                        otherCell->cellTypeData.injector.genomeSize = injector.genomeSize;
                    }

                    injection = true;
                    otherCell->releaseLock();
                });
        } break;

        case InjectorMode_InjectOnlyEmptyCells: {
            data.cellMap.executeForEach(
                cell->pos, cudaSimulationParameters.injectorInjectionRadius.value[cell->color], cell->detached, [&](Cell* const& otherCell) {
                    if (cell == otherCell) {
                        return;
                    }
                    if (otherCell->cellType != CellType_Constructor && otherCell->cellType != CellType_Injector) {
                        return;
                    }
                    auto otherGenomeSize = otherCell->cellType == CellType_Constructor ? otherCell->cellTypeData.constructor.genomeSize
                                                                                               : otherCell->cellTypeData.injector.genomeSize;
                    if (otherGenomeSize > Const::GenomeHeaderSize) {
                        return;
                    }
                    if (!otherCell->tryLock()) {
                        return;
                    }
                    auto targetGenome = data.objects.heap.getRawSubArray(injector.genomeSize);
                    for (int i = 0; i < injector.genomeSize; ++i) {
                        targetGenome[i] = injector.genome[i];
                    }
                    if (otherCell->cellType == CellType_Constructor) {
                        otherCell->cellTypeData.constructor.genome = targetGenome;
                        otherCell->cellTypeData.constructor.genomeSize = injector.genomeSize;
                        otherCell->cellTypeData.constructor.numInheritedGenomeNodes = 0;
                    } else {
                        otherCell->cellTypeData.injector.genome = targetGenome;
                        otherCell->cellTypeData.injector.genomeSize = injector.genomeSize;
                    }
                    match = true;
                    injection = true;
                    otherCell->releaseLock();
                });
        } break;
        }

        if (match) {
            statistics.incNumInjectionActivities(cell->color);
            if (injection) {
                statistics.incNumCompletedInjections(cell->color);
                injector.counter = 0;
            } else {
                ++injector.counter;
            }
            cell->signal.channels[0] = 1;
        } else {
            injector.counter = 0;
            cell->signal.channels[0] = 0;
        }
    }
}

__inline__ __device__ int InjectorProcessor::countAndTrackDefenderCells(SimulationStatistics& statistics, Cell* cell)
{
    int result = 0;
    for (int i = 0; i < cell->numConnections; ++i) {
        auto connectedCell = cell->connections[i].cell;
        if (connectedCell->cellType == CellType_Defender && connectedCell->cellTypeData.defender.mode == DefenderMode_DefendAgainstInjector) {
            statistics.incNumDefenderActivities(connectedCell->color);
            ++result;
        }
    }
    return result;
}
