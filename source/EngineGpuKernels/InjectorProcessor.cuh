#pragma once

#include "CellFunctionProcessor.cuh"
#include "SimulationData.cuh"
#include "sm_60_atomic_functions.h"

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
    auto activity = CellFunctionProcessor::calcInputActivity(cell);

    if (abs(activity.channels[0]) < cudaSimulationParameters.cellFunctionInjectorActivityThreshold) {
        return;
    }

    auto& injector = cell->cellFunctionData.injector;

    bool match = false;
    bool injection = false;

    switch (injector.mode) {
    case InjectorMode_InjectAll: {
        data.cellMap.executeForEach(cell->pos, cudaSimulationParameters.cellFunctionInjectorRadius[cell->color], cell->detached, [&](Cell* const& otherCell) {
            if (cell == otherCell) {
                return;
            }
            if (otherCell->cellFunction != CellFunction_Constructor && otherCell->cellFunction != CellFunction_Injector) {
                return;
            }
            //if (otherCell->livingState == LivingState_UnderConstruction) {
            //    return;
            //}
            if (otherCell->cellFunctionData.constructor.genomeReadPosition != 0) {
                return;
            }
            match = true;
            auto injectorDuration = cudaSimulationParameters.cellFunctionInjectorDurationColorMatrix[cell->color][otherCell->color];

            auto numDefenderCells = countAndTrackDefenderCells(statistics, otherCell);
            float defendStrength =
                numDefenderCells == 0 ? 1.0f : powf(cudaSimulationParameters.cellFunctionDefenderAgainstInjectorStrength[cell->color], numDefenderCells);
            injectorDuration = toInt(toFloat(injectorDuration) * defendStrength);

            if (injector.counter < injectorDuration) {
                return;
            }

            auto targetGenome = data.objects.auxiliaryData.getAlignedSubArray(injector.genomeSize);
            for (int i = 0; i < injector.genomeSize; ++i) {
                targetGenome[i] = injector.genome[i];
            }

            if (otherCell->cellFunction == CellFunction_Constructor) {
                otherCell->cellFunctionData.constructor.genome = targetGenome;
                otherCell->cellFunctionData.constructor.genomeSize = injector.genomeSize;
            } else {
                otherCell->cellFunctionData.injector.genome = targetGenome;
                otherCell->cellFunctionData.injector.genomeSize = injector.genomeSize;
            }

            atomicAdd(&otherCell->activity.channels[6], 1.0f);
            injection = true;
        });
    } break;

    case InjectorMode_InjectOnlyUnderConstruction: {
        for (int i = 0; i < cell->numConnections; ++i) {
            auto& connectedCell = cell->connections[i].cell;
            for (int j = 0; j < connectedCell->numConnections; ++j) {
                auto connectedConnectedCell = connectedCell->connections[j].cell;
                if (connectedConnectedCell->livingState == LivingState_UnderConstruction
                    && connectedConnectedCell->cellFunctionData.constructor.genomeReadPosition == 0) {
                    auto targetGenome = data.objects.auxiliaryData.getAlignedSubArray(injector.genomeSize);
                    for (int i = 0; i < injector.genomeSize; ++i) {
                        targetGenome[i] = injector.genome[i];
                    }
                    if (connectedConnectedCell->cellFunction == CellFunction_Constructor) {
                        connectedConnectedCell->cellFunctionData.constructor.genome = targetGenome;
                        connectedConnectedCell->cellFunctionData.constructor.genomeSize = injector.genomeSize;
                    } else {
                        connectedConnectedCell->cellFunctionData.injector.genome = targetGenome;
                        connectedConnectedCell->cellFunctionData.injector.genomeSize = injector.genomeSize;
                    }
                    atomicAdd(&connectedConnectedCell->activity.channels[6], 1.0f);
                    match = true;
                    injection = true;
                }
            }
        }
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
        activity.channels[0] = 1.0;
    } else {
        injector.counter = 0;
        activity.channels[0] = 0.0;
    }

    CellFunctionProcessor::setActivity(cell, activity);
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
