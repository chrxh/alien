#pragma once

#include "TOs.cuh"
#include "Base.cuh"
#include "Map.cuh"
#include "ObjectFactory.cuh"
#include "SpotCalculator.cuh"

class CellFunctionProcessor
{
public:
    __inline__ __device__ static void collectCellFunctionOperations(SimulationData& data);
    __inline__ __device__ static void resetFetchedActivities(SimulationData& data);
    __inline__ __device__ static void aging(SimulationData& data);

    __inline__ __device__ static Activity calcInputActivity(Cell* cell, int& inputExecutionOrderNumber);
    __inline__ __device__ static void setActivity(Cell* cell, Activity const& newActivity);

private:
    __inline__ __device__ static int calcInputExecutionOrder(Cell* cell);   //returns -1 if no input has been found
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
        if (cell->cellFunction != Enums::CellFunction_None && cell->executionOrderNumber == executionOrderNumber && cell->activationTime == 0) {
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

__inline__ __device__ void CellFunctionProcessor::aging(SimulationData& data)
{
    auto const partition = calcAllThreadsPartition(data.objects.cellPointers.getNumEntries());
    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = data.objects.cellPointers.at(index);
        if (cell->barrier) {
            continue;
        }

        auto color = calcMod(cell->color, 7);
        auto transitionDuration = SpotCalculator::calcColorTransitionDuration(color, data, cell->absPos);
        ++cell->age;
        if (transitionDuration > 0 && cell->age > transitionDuration) {
            auto targetColor = SpotCalculator::calcColorTransitionTargetColor(color, data, cell->absPos);
            cell->color = targetColor;
            cell->age = 0;
        }
        if (cell->constructionState == Enums::ConstructionState_Finished && cell->activationTime > 0) {
            --cell->activationTime;
        }
    }
}

__inline__ __device__ Activity CellFunctionProcessor::calcInputActivity(Cell* cell, int& inputExecutionOrderNumber)
{
    Activity result;

    for (int i = 0; i < MAX_CHANNELS; ++i) {
        result.channels[i] = 0;
    }

    inputExecutionOrderNumber = calcInputExecutionOrder(cell);
    if (inputExecutionOrderNumber == -1) {
        return result;
    }

    for (int i = 0; i < cell->numConnections; ++i) {
        auto connectedCell = cell->connections[i].cell;
        if (connectedCell->outputBlocked || connectedCell->constructionState || connectedCell->cellFunction == Enums::CellFunction_None) {
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

__inline__ __device__ int CellFunctionProcessor::calcInputExecutionOrder(Cell* cell)
{
    if (cell->inputBlocked || cell->constructionState) {
        return -1;
    }
    int result = -cudaSimulationParameters.cellMaxExecutionOrderNumbers;
    for (int i = 0; i < cell->numConnections; ++i) {
        auto connectedCell = cell->connections[i].cell;
        if (connectedCell->outputBlocked || connectedCell->constructionState) {
            continue;
        }
        auto otherExecutionOrderNumber = connectedCell->executionOrderNumber;
        if (otherExecutionOrderNumber > cell->executionOrderNumber) {
            otherExecutionOrderNumber -= cudaSimulationParameters.cellMaxExecutionOrderNumbers;
        }
        if (otherExecutionOrderNumber > result && otherExecutionOrderNumber != cell->executionOrderNumber) {
            result = otherExecutionOrderNumber;
        }
    }

    if (result == -cudaSimulationParameters.cellMaxExecutionOrderNumbers) {
        return -1;
    }
    if (result < 0) {
        result += cudaSimulationParameters.cellMaxExecutionOrderNumbers;
    }
    return result;
}
