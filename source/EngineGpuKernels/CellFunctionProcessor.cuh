#pragma once

#include "TOs.cuh"
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
    __inline__ __device__ static void resetFetchedActivities(SimulationData& data);

    __inline__ __device__ static Activity calcInputActivity(Cell* cell);
    __inline__ __device__ static void setActivity(Cell* cell, Activity const& newActivity);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__inline__ __device__ void CellFunctionProcessor::collectCellFunctionOperations(SimulationData& data)
{
    auto& cells = data.objects.cellPointers;
    auto partition = calcAllThreadsPartition(cells.getNumEntries());

    auto maxExecutionOrderNumber = cudaSimulationParameters.cellMaxExecutionOrderNumbers;
    auto executionOrderNumber = data.timestep % maxExecutionOrderNumber;
    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cells.at(index);
        if (cell->cellFunction != Enums::CellFunction_None && cell->executionOrderNumber == executionOrderNumber) {
            data.cellFunctionOperations[cell->cellFunction].tryAddEntry(CellFunctionOperation{cell});
        }
    }
}

__inline__ __device__ void CellFunctionProcessor::resetFetchedActivities(SimulationData& data)
{
    auto& cells = data.objects.cellPointers;
    auto partition = calcAllThreadsPartition(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cells.at(index);
        if (cell->activityFetched == 1) {
            for (int i = 0; i < MAX_CHANNELS; ++i) {
                cell->activity.channels[i] = 0;
            }
        }
    }
}

__inline__ __device__ Activity CellFunctionProcessor::calcInputActivity(Cell* cell)
{
    Activity result;

    for (int i = 0; i < MAX_CHANNELS; ++i) {
        result.channels[i] = 0;
    }

    if (cell->inputBlocked || cell->underConstruction) {
        return result;
    }
    int inputExecutionOrderNumber = -cudaSimulationParameters.cellMaxExecutionOrderNumbers;
    for (int i = 0; i < cell->numConnections; ++i) {
        auto connectedCell = cell->connections[i].cell;
        if (connectedCell->outputBlocked || connectedCell->underConstruction) {
            continue;
        }
        auto otherExecutionOrderNumber = connectedCell->executionOrderNumber;
        if (otherExecutionOrderNumber > cell->executionOrderNumber) {
            otherExecutionOrderNumber -= cudaSimulationParameters.cellMaxExecutionOrderNumbers;
        }
        if (otherExecutionOrderNumber > inputExecutionOrderNumber) {
            inputExecutionOrderNumber = otherExecutionOrderNumber;
        }
    }

    if (inputExecutionOrderNumber == -cudaSimulationParameters.cellMaxExecutionOrderNumbers) {
        return result;
    }
    if (inputExecutionOrderNumber < 0) {
        inputExecutionOrderNumber += cudaSimulationParameters.cellMaxExecutionOrderNumbers;
    }
    for (int i = 0; i < cell->numConnections; ++i) {
        auto connectedCell = cell->connections[i].cell;
        if (connectedCell->outputBlocked || connectedCell->underConstruction) {
            continue;
        }
        if (connectedCell->executionOrderNumber == inputExecutionOrderNumber) {
            for (int i = 0; i < MAX_CHANNELS; ++i) {
                result.channels[i] += connectedCell->activity.channels[i];
                atomicExch(&connectedCell->activityFetched, 1);
            }
        }
    }
    return result;
}

__inline__ __device__ void CellFunctionProcessor::setActivity(Cell* cell, Activity const& newActivity)
{
    auto changes = 0.0f;
    for (int i = 0; i < MAX_CHANNELS; ++i) {
        changes += abs(newActivity.channels[i] - cell->activity.channels[i]);
        cell->activity.channels[i] = newActivity.channels[i];
    }
    cell->activity = newActivity;
}
