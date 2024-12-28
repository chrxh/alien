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
    auto& operations = data.cellFunctionOperations[CellFunction_Injector];
    auto partition = calcAllThreadsPartition(operations.getNumEntries());
    for (int i = partition.startIndex; i <= partition.endIndex; ++i) {
        processCell(data, statistics, operations.at(i).cell);
    }
}

__inline__ __device__ void InjectorProcessor::processCell(SimulationData& data, SimulationStatistics& statistics, Cell* cell)
{
    if (abs(cell->signal.channels[0]) >= cudaSimulationParameters.cellFunctionInjectorSignalThreshold) {

        auto& injector = cell->cellFunctionData.injector;

        bool match = false;
        bool injection = false;

        switch (injector.mode) {
        case InjectorMode_InjectAll: {
            data.cellMap.executeForEach(
                cell->pos, cudaSimulationParameters.cellFunctionInjectorRadius[cell->color], cell->detached, [&](Cell* const& otherCell) {
                    if (cell == otherCell) {
                        return;
                    }
                    if (otherCell->cellFunction != CellFunction_Constructor && otherCell->cellFunction != CellFunction_Injector) {
                        return;
                    }
                    //if (otherCell->livingState == LivingState_UnderConstruction) {
                    //    return;
                    //}
                    if (otherCell->cellFunctionData.constructor.genomeCurrentNodeIndex != 0) {
                        return;
                    }
                    if (!otherCell->tryLock()) {
                        return;
                    }
                    match = true;
                    auto injectorDuration = cudaSimulationParameters.cellFunctionInjectorDurationColorMatrix[cell->color][otherCell->color];

                    auto numDefenderCells = countAndTrackDefenderCells(statistics, otherCell);
                    float defendStrength = numDefenderCells == 0
                        ? 1.0f
                        : powf(cudaSimulationParameters.cellFunctionDefenderAgainstInjectorStrength[cell->color], numDefenderCells);
                    injectorDuration = toInt(toFloat(injectorDuration) * defendStrength);
                    if (injector.counter < injectorDuration) {
                        otherCell->releaseLock();
                        return;
                    }

                    auto targetGenome = data.objects.auxiliaryData.getAlignedSubArray(injector.genomeSize);
                    for (int i = 0; i < injector.genomeSize; ++i) {
                        targetGenome[i] = injector.genome[i];
                    }

                    if (otherCell->cellFunction == CellFunction_Constructor) {
                        otherCell->cellFunctionData.constructor.genome = targetGenome;
                        otherCell->cellFunctionData.constructor.genomeSize = injector.genomeSize;
                        otherCell->cellFunctionData.constructor.numInheritedGenomeNodes = 0;
                    } else {
                        otherCell->cellFunctionData.injector.genome = targetGenome;
                        otherCell->cellFunctionData.injector.genomeSize = injector.genomeSize;
                    }

                    injection = true;
                    otherCell->releaseLock();
                });
        } break;

        case InjectorMode_InjectOnlyEmptyCells: {
            data.cellMap.executeForEach(
                cell->pos, cudaSimulationParameters.cellFunctionInjectorRadius[cell->color], cell->detached, [&](Cell* const& otherCell) {
                    if (cell == otherCell) {
                        return;
                    }
                    if (otherCell->cellFunction != CellFunction_Constructor && otherCell->cellFunction != CellFunction_Injector) {
                        return;
                    }
                    auto otherGenomeSize = otherCell->cellFunction == CellFunction_Constructor ? otherCell->cellFunctionData.constructor.genomeSize
                                                                                               : otherCell->cellFunctionData.injector.genomeSize;
                    if (otherGenomeSize > Const::GenomeHeaderSize) {
                        return;
                    }
                    if (!otherCell->tryLock()) {
                        return;
                    }
                    auto targetGenome = data.objects.auxiliaryData.getAlignedSubArray(injector.genomeSize);
                    for (int i = 0; i < injector.genomeSize; ++i) {
                        targetGenome[i] = injector.genome[i];
                    }
                    if (otherCell->cellFunction == CellFunction_Constructor) {
                        otherCell->cellFunctionData.constructor.genome = targetGenome;
                        otherCell->cellFunctionData.constructor.genomeSize = injector.genomeSize;
                        otherCell->cellFunctionData.constructor.numInheritedGenomeNodes = 0;
                    } else {
                        otherCell->cellFunctionData.injector.genome = targetGenome;
                        otherCell->cellFunctionData.injector.genomeSize = injector.genomeSize;
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
        if (connectedCell->cellFunction == CellFunction_Defender && connectedCell->cellFunctionData.defender.mode == DefenderMode_DefendAgainstInjector) {
            statistics.incNumDefenderActivities(connectedCell->color);
            ++result;
        }
    }
    return result;
}
