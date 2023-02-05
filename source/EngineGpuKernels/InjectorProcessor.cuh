#pragma once

#include "CellFunctionProcessor.cuh"
#include "SimulationData.cuh"
#include "sm_60_atomic_functions.h"

class InjectorProcessor
{
public:
    __inline__ __device__ static void process(SimulationData& data, SimulationResult& result);

private:
    __inline__ __device__ static void processCell(SimulationData& data, SimulationResult& result, Cell* cell);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__device__ __inline__ void InjectorProcessor::process(SimulationData& data, SimulationResult& result)
{
    auto& operations = data.cellFunctionOperations[CellFunction_Injector];
    auto partition = calcAllThreadsPartition(operations.getNumEntries());
    for (int i = partition.startIndex; i <= partition.endIndex; ++i) {
        processCell(data, result, operations.at(i).cell);
    }
}

__inline__ __device__ void InjectorProcessor::processCell(SimulationData& data, SimulationResult& result, Cell* cell)
{
    int inputExecutionOrderNumber;
    auto activity = CellFunctionProcessor::calcInputActivity(cell, inputExecutionOrderNumber);

    if (abs(activity.channels[0]) < cudaSimulationParameters.cellFunctionInjectorActivityThreshold) {
        return;
    }

    auto& injector = cell->cellFunctionData.injector;

    bool match = false;
    bool injection = false;
    if (injector.mode == InjectorMode_InjectAll) {
        data.cellMap.executeForEach(cell->absPos, cudaSimulationParameters.cellFunctionInjectorRadius, cell->detached, [&](Cell* const& otherCell) {
            if (cell == otherCell) {
                return;
            }
            if (otherCell->cellFunction != CellFunction_Constructor && otherCell->cellFunction != CellFunction_Injector) {
                return;
            }
            match = true;
            auto injectorDuration = cudaSimulationParameters.cellFunctionInjectorDurationColorMatrix[cell->color][otherCell->color];
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
            injection = true;
        });
    } else {    //InjectorMode_InjectOnlyUnderConstruction
        for (int i = 0; i < cell->numConnections; ++i) {
            auto& connectedCell = cell->connections[i].cell;
            for (int j = 0; j < connectedCell->numConnections; ++j) {
                auto connectedConnectedCell = connectedCell->connections[j].cell;
                if (connectedConnectedCell->livingState == LivingState_UnderConstruction) {
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
                    match = true;
                    injection = true;
                }
            }
        }
    }

    if (match) {
        if (injection) {
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
