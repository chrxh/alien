#pragma once

#include "EngineInterface/CellTypeConstants.h"

#include "Object.cuh"
#include "SimulationData.cuh"
#include "SignalProcessor.cuh"
#include "SimulationStatistics.cuh"

class MuscleProcessor
{
public:
    __inline__ __device__ static void process(SimulationData& data, SimulationStatistics& statistics);

private:
    __inline__ __device__ static void processCell(SimulationData& data, SimulationStatistics& statistics, Cell* cell);

    //__inline__ __device__ static void movement(SimulationData& data, SimulationStatistics& statistics, Cell* cell);
    //__inline__ __device__ static void contractionExpansion(SimulationData& data, SimulationStatistics& statistics, Cell* cell);
    __inline__ __device__ static void bending(SimulationData& data, SimulationStatistics& statistics, Cell* cell);

    //__inline__ __device__ static int getConnectionIndex(Cell* cell, Cell* otherCell);
    //__inline__ __device__ static bool hasTriangularConnection(Cell* cell, Cell* otherCell);
    //__inline__ __device__ static float getTruncatedUnitValue(Signal const& signal, int channel = 0);
    __inline__ __device__ static void radiate(SimulationData& data, Cell* cell);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/
__device__ __inline__ void MuscleProcessor::process(SimulationData& data, SimulationStatistics& statistics)
{
    auto& operations = data.cellTypeOperations[CellType_Muscle];
    auto partition = calcAllThreadsPartition(operations.getNumEntries());
    for (int i = partition.startIndex; i <= partition.endIndex; ++i) {
        processCell(data, statistics, operations.at(i).cell);
    }
}

__device__ __inline__ void MuscleProcessor::processCell(SimulationData& data, SimulationStatistics& statistics, Cell* cell)
{
    cell->cellTypeData.muscle.lastMovementX = 0;
    cell->cellTypeData.muscle.lastMovementY = 0;

    switch (cell->cellTypeData.muscle.mode) {
    //case MuscleMode_Movement: {
    //    movement(data, statistics, cell);
    //} break;
    //case MuscleMode_ContractionExpansion: {
    //    contractionExpansion(data, statistics, cell);
    //} break;
    case MuscleMode_Bending: {
        bending(data, statistics, cell);
    } break;
    }
}


//__device__ __inline__ void MuscleProcessor::movement(SimulationData& data, SimulationStatistics& statistics, Cell* cell)
//{
//    if (abs(cell->signal.channels[0]) < NEAR_ZERO) {
//        return;
//    }
//    if (!cell->tryLock()) {
//        return;
//    }
//
//    auto direction = SignalProcessor::calcReferenceDirection(data, cell);
//    auto acceleration = cudaSimulationParameters.cellTypeMuscleMovementAcceleration[cell->color];
//    float angle = max(-0.5f, min(0.5f, cell->signal.channels[3])) * 360.0f;
//    direction = Math::normalized(Math::rotateClockwise(direction, angle)) * acceleration * getTruncatedUnitValue(cell->signal);
//    cell->vel += direction;
//    cell->cellTypeData.muscle.lastMovementX = direction.x;
//    cell->cellTypeData.muscle.lastMovementY = direction.y;
//    cell->releaseLock();
//    statistics.incNumMuscleActivities(cell->color);
//    radiate(data, cell);
//}
//
//__device__ __inline__ void MuscleProcessor::contractionExpansion(SimulationData& data, SimulationStatistics& statistics, Cell* cell)
//{
//    if (abs(cell->signal.channels[0]) < NEAR_ZERO) {
//        return;
//    }
//    if (!cell->tryLock()) {
//        return;
//    }
//    auto const minDistance = cudaSimulationParameters.cellMinDistance * 1.2f;
//    auto const maxDistance = max(cudaSimulationParameters.cellMaxBindingDistance[cell->color] * 0.5f, minDistance);
//    for (int i = 0; i < cell->numConnections; ++i) {
//        auto& connection = cell->connections[i];
//        //if (connection.cell->executionOrderNumber == cell->inputExecutionOrderNumber) {
//            if (!connection.cell->tryLock()) {
//                continue;
//            }
//            auto newDistance =
//                connection.distance + cudaSimulationParameters.cellTypeMuscleContractionExpansionDelta[cell->color] * getTruncatedUnitValue(cell->signal);
//            if (cell->signal.channels[0] > 0 && newDistance >= maxDistance) {
//                continue;
//            }
//            if (cell->signal.channels[0] < 0 && newDistance <= minDistance) {
//                continue;
//            }
//            connection.distance = newDistance;
//
//            auto otherIndex = getConnectionIndex(connection.cell, cell);
//            connection.cell->connections[otherIndex].distance = newDistance;
//            connection.cell->releaseLock();
//        //}
//    }
//    cell->releaseLock();
//    statistics.incNumMuscleActivities(cell->color);
//    radiate(data, cell);
//}

__inline__ __device__ void MuscleProcessor::bending(SimulationData& data, SimulationStatistics& statistics, Cell* cell)
{
    auto& muscle = cell->cellTypeData.muscle;
    auto& bending = muscle.modeData.bending;

    if (cell->numConnections != 2) {
        return;
    }
    if (bending.forwardVel < NEAR_ZERO || bending.backwardVel < NEAR_ZERO) {
        return;
    }
    if (SignalProcessor::isAutoTriggered(data, cell, 10)) {

        if (bending.offsetCounter < bending.offset) {
            ++bending.offsetCounter;
            return;
        }

        auto bendForwardVel = bending.forwardVel;
        auto bendBackwardVel = bending.backwardVel;
        auto orientation = Math::normalizedAngle(Math::subtractAngle(cell->absAngleToConnection0, muscle.frontAngle), -180.0f);

        auto bendForwardSteps = toInt(2.0f / bendForwardVel);
        auto bendBackwardSteps = toInt(2.0f / bendBackwardVel);

        auto cycle = (bendForwardSteps + bendBackwardSteps) * 2;
        auto currentStep = bending.currentStep % cycle;
        bool isForward = currentStep < bendForwardSteps || currentStep >= (cycle - bendForwardSteps);
        auto angleDelta = isForward ? bending.maxAngleDeviation / toFloat(bendForwardSteps) : bending.maxAngleDeviation / toFloat(bendBackwardSteps);
        auto bendingVel = isForward ? bendForwardVel : bendBackwardVel;

        if (orientation < 0) {
            isForward = !isForward;
        }
        if (isForward) {
            if (cell->connections[1].angleFromPrevious > 60.0f + angleDelta) {
                cell->connections[0].angleFromPrevious += angleDelta;
                cell->connections[1].angleFromPrevious -= angleDelta;
            } else {
                bending.currentStep = bendForwardSteps - 1;
            }
        } else {
            if (cell->connections[0].angleFromPrevious > 60.0f + angleDelta) {
                cell->connections[0].angleFromPrevious -= angleDelta;
                cell->connections[1].angleFromPrevious += angleDelta;
            } else {
                bending.currentStep = bendForwardSteps + 2 * bendBackwardSteps - 1;
            }
        }

        auto delta = Math::normalized(data.cellMap.getCorrectedDirection(cell->connections[0].cell->pos - cell->pos));
        if (isForward) {
            Math::rotateQuarterCounterClockwise(delta);
        } else {
            Math::rotateQuarterClockwise(delta);
        }
        auto acceleration = delta * sqrt(sqrt(bendingVel)) * cudaSimulationParameters.cellTypeMuscleBendingAcceleration[cell->color] / 20;
        atomicAdd(&cell->connections[0].cell->vel.x, acceleration.x);
        atomicAdd(&cell->connections[0].cell->vel.y, acceleration.y);

        bending.currentStep = (bending.currentStep + 1) % cycle; 


        statistics.incNumMuscleActivities(cell->color);
        radiate(data, cell);
    }
}

//__inline__ __device__ int MuscleProcessor::getConnectionIndex(Cell* cell, Cell* otherCell)
//{
//    for (int i = 0; i < cell->numConnections; ++i) {
//        if (cell->connections[i].cell == otherCell) {
//            return i;
//        }
//    }
//    return 0;
//}

//__inline__ __device__ bool MuscleProcessor::hasTriangularConnection(Cell* cell, Cell* otherCell)
//{
//    for (int i = 0; i < cell->numConnections; ++i) {
//        auto connectedCell = cell->connections[i].cell;
//        if (connectedCell == otherCell) {
//            continue;
//        }
//        for (int j = 0; j < connectedCell->numConnections; ++j) {
//            auto connectedConnectedCell = connectedCell->connections[j].cell;
//            if (connectedConnectedCell == otherCell) {
//                return true;
//            }
//        }
//    }
//    return false;
//}
//
//__inline__ __device__ float MuscleProcessor::getTruncatedUnitValue(Signal const& signal, int channel)
//{
//    return max(-0.3f, min(0.3f, signal.channels[channel])) / 0.3f;
//}
//
__inline__ __device__ void MuscleProcessor::radiate(SimulationData& data, Cell* cell)
{
    auto cellTypeMuscleEnergyCost = cudaSimulationParameters.cellTypeMuscleEnergyCost[cell->color];
    if (cellTypeMuscleEnergyCost > 0) {
        RadiationProcessor::radiate(data, cell, cellTypeMuscleEnergyCost);
    }
}
