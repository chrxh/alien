#pragma once

#include "AccessTOs.cuh"
#include "Base.cuh"
#include "CellConnectionProcessor.cuh"
#include "Map.cuh"
#include "ObjectFactory.cuh"
#include "Physics.cuh"
#include "SpotCalculator.cuh"
#include "cuda_runtime_api.h"
#include "sm_60_atomic_functions.h"

class CellFunctionProcessor
{
public:
    __inline__ __device__ static void collectCellFunctionOperations(SimulationData& data);

    __inline__ __device__ static void calcInputActivity(Cell* cell, bool& inputAvailable, Activity& result);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__inline__ __device__ void CellFunctionProcessor::collectCellFunctionOperations(SimulationData& data)
{
    auto& cells = data.objects.cellPointers;
    auto partition = calcAllThreadsPartition(cells.getNumEntries());

    auto maxExecutionOrderNumber = cudaSimulationParameters.cellMaxExecutionOrderNumber;
    auto executionOrderNumber = data.timestep % maxExecutionOrderNumber;
    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cells.at(index);
        if (cell->cellFunction != Enums::CellFunction_None && cell->executionOrderNumber == executionOrderNumber) {
            data.cellFunctionOperations[cell->cellFunction].tryAddEntry(CellFunctionOperation{cell});
        }
    }
}

__inline__ __device__ void CellFunctionProcessor::calcInputActivity(Cell* cell, bool& inputAvailable, Activity& result)
{
    int inputExecutionOrderNumber = -cudaSimulationParameters.cellMaxExecutionOrderNumber;
    for (int i = 0; i < cell->numConnections; ++i) {
        auto connectedCell = cell->connections[i].cell;
        if (connectedCell->inputBlocked) {
            continue;
        }
        auto otherExecutionOrderNumber = connectedCell->executionOrderNumber;
        if (otherExecutionOrderNumber > cell->executionOrderNumber) {
            otherExecutionOrderNumber -= cudaSimulationParameters.cellMaxExecutionOrderNumber;
        }
        if (otherExecutionOrderNumber > inputExecutionOrderNumber) {
            inputExecutionOrderNumber = otherExecutionOrderNumber;
        }
    }

    if (inputExecutionOrderNumber == -cudaSimulationParameters.cellMaxExecutionOrderNumber) {
        inputAvailable = false;
        return;
    }

    inputAvailable = true;
    if (inputExecutionOrderNumber < 0) {
        inputExecutionOrderNumber += cudaSimulationParameters.cellMaxExecutionOrderNumber;
    }
    for (int i = 0; i < MAX_CHANNELS; ++i) {
        result.channels[i] = 0;
    }

    for (int i = 0; i < cell->numConnections; ++i) {
        auto connectedCell = cell->connections[i].cell;
        if (connectedCell->inputBlocked) {
            continue;
        }
        if (connectedCell->executionOrderNumber == inputExecutionOrderNumber) {
            for (int i = 0; i < MAX_CHANNELS; ++i) {
                result.channels[i] += connectedCell->activity.channels[i];
            }
        }
    }
}
