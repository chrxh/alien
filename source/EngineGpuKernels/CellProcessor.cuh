#pragma once

#include <EngineInterface/CellTypeConstants.h>

#include "cuda_runtime_api.h"
#include "sm_60_atomic_functions.h"

#include "Base.cuh"
#include "CellConnectionProcessor.cuh"
#include "ConstructorHelper.cuh"
#include "Map.cuh"
#include "MuscleProcessor.cuh"
#include "ObjectFactory.cuh"
#include "ParameterCalculator.cuh"
#include "Physics.cuh"
#include "EnergyParticleProcessor.cuh"
#include "TO.cuh"

class CellProcessor
{
public:
    __inline__ __device__ static void init(SimulationData& data);
    __inline__ __device__ static void updateMap(SimulationData& data);
    __inline__ __device__ static void clearDensityMap(SimulationData& data);
    __inline__ __device__ static void fillDensityMap(SimulationData& data);

    __inline__ __device__ static void calcFluidForces_reconnectCells_correctOverlap(SimulationData& data);
    __inline__ __device__ static void calcCollisions_reconnectCells_correctOverlap(SimulationData& data);
    __inline__ __device__ static void checkForces(SimulationData& data);
    __inline__ __device__ static void applyForces(SimulationData& data);  //prerequisite: data from calcCollisions_reconnectCells_correctOverlap

    __inline__ __device__ static void calcConnectionForces(SimulationData& data, bool calcAngularForces);
    __inline__ __device__ static void checkConnections(SimulationData& data);
    __inline__ __device__ static void verletPositionUpdate(SimulationData& data);
    __inline__ __device__ static void verletVelocityUpdate(SimulationData& data);

    __inline__ __device__ static void aging(SimulationData& data);
    __inline__ __device__ static void cellStateTransition_calcFutureState(SimulationData& data);
    __inline__ __device__ static void cellStateTransition_applyNextState(SimulationData& data);
    __inline__ __device__ static void frontAngleUpdate_calcFutureValue(SimulationData& data);
    __inline__ __device__ static void frontAngleUpdate_applyFutureValue(SimulationData& data);

    __inline__ __device__ static void applyInnerFriction(SimulationData& data);
    __inline__ __device__ static void applyFriction(SimulationData& data);

    __inline__ __device__ static void radiation(SimulationData& data);
    __inline__ __device__ static void decay(SimulationData& data);

    __inline__ __device__ static void resetDensity(SimulationData& data);

    __inline__ __device__ static void updateCellEvents(SimulationData& data);

    __inline__ __device__ static void performEnergyFlow(SimulationData& data);

private:
    static auto constexpr MaxBarrierCellsForCollision = 10;
    static auto constexpr FrontAngleId_NoUpdate = VALUE_NOT_SET_FLOAT;

    __inline__ __device__ static float getInitialAngelSpan(Cell* cell, int connectionIndex1, int connectionIndex2);
    __inline__ __device__ static float getInitialAngelSpan(Cell* cell, Cell* connectedCell1, Cell* connectedCell2);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__inline__ __device__ void CellProcessor::init(SimulationData& data)
{
    auto& cells = data.objects.cells;
    auto partition = calcSystemThreadPartition(cells.getNumEntries());
    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& cell = cells.at(index);

        cell->shared1 = {0, 0};
        cell->nextCell = nullptr;
        cell->tempValue.as_uint64 = 0;
    }
}

__inline__ __device__ void CellProcessor::updateMap(SimulationData& data)
{
    auto const partition = calcBlockPartition(data.objects.cells.getNumEntries());
    Cell** cellPointers = &data.objects.cells.at(partition.startIndex);
    data.cellMap.set_block(partition.numElements(), cellPointers);
}

__inline__ __device__ void CellProcessor::clearDensityMap(SimulationData& data)
{
    data.preprocessedSimulationData.densityMap.clear();
}

__inline__ __device__ void CellProcessor::fillDensityMap(SimulationData& data)
{
    auto const partition = calcSystemThreadPartition(data.objects.cells.getNumEntries());
    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto cell = data.objects.cells.at(index);
        if (cell->cellType == CellType_Free) {
            data.preprocessedSimulationData.densityMap.addFreeCell(cell);
        } else if (cell->cellType == CellType_Structure) {
            data.preprocessedSimulationData.densityMap.addStructureCell(cell);
        }
    }
}

namespace
{
    __inline__ __device__ float calcKernel(float q)
    {
        float result;
        if (q < 1) {
            result = 2.0f / 3.0f - q * q + 0.5f * q * q * q;
        } else if (q < 2) {
            result = 2.0f - q;
            result = result * result * result / 6;
        } else {
            result = 0;
        }
        result *= 3.0f / (2.0f * Const::PI);
        return result;
    }

    __inline__ __device__ float calcKernel_d(float q)
    {
        float result;
        if (q < 1) {
            result = -2 * q + 3.0f / 2.0f * q * q;
        } else if (q < 2) {
            result = -0.5f * (2.0f - q) * (2.0f - q);
        } else {
            result = 0;
        }
        result *= 3.0f / (2.0f * Const::PI);
        return result;
    }
}

__inline__ __device__ void CellProcessor::calcFluidForces_reconnectCells_correctOverlap(SimulationData& data)
{
    auto& cells = data.objects.cells;
    auto blockPartition = calcBlockPartition(cells.getNumEntries());
    auto const& smoothingLength = cudaSimulationParameters.smoothingLength.value;

    for (int cellIndex = blockPartition.startIndex; cellIndex <= blockPartition.endIndex; ++cellIndex) {
        auto& cell = cells.at(cellIndex);

        __shared__ float2 F_pressure;
        __shared__ float2 F_viscosity;
        __shared__ float2 cellPosDelta;
        __shared__ float density;
        __shared__ float cellMaxBindingEnergy;
        __shared__ float cellFusionVelocity;

        __shared__ int scanLength;
        __shared__ int2 cellPosInt;

        __shared__ Cell* fixedCells[MaxBarrierCellsForCollision];
        __shared__ int numFixedCells;

        if (threadIdx.x == 0) {
            F_pressure = {0, 0};
            F_viscosity = {0, 0};
            cellPosDelta = {0, 0};
            density = 0;
            cellMaxBindingEnergy = ParameterCalculator::calcParameter(cudaSimulationParameters.cellMaxBindingEnergy, data, cell->pos);
            cellFusionVelocity = ParameterCalculator::calcParameter(cudaSimulationParameters.cellFusionVelocity, data, cell->pos);

            int radiusInt = ceilf(smoothingLength * 2);
            scanLength = radiusInt * 2 + 1;
            cellPosInt = {floorInt(cell->pos.x) - radiusInt, floorInt(cell->pos.y) - radiusInt};

            numFixedCells = 0;
        }
        __syncthreads();

        int2 scanPos{cellPosInt.x + (toInt(threadIdx.x) % scanLength), cellPosInt.y + (toInt(threadIdx.x) / scanLength)};
        data.cellMap.correctPosition(scanPos);
        auto otherCell = data.cellMap.getFirst(scanPos);
        for (int level = 0; level < MaxBarrierCellsForCollision; ++level) {
            if (!otherCell) {
                break;
            }
            auto posDelta = cell->pos - otherCell->pos;
            data.cellMap.correctDirection(posDelta);
            auto adaptedDistance = Math::length(posDelta);
            auto origDistance = adaptedDistance;
            if (cell->isSameCreature(otherCell) && (cell->numConnections < 3 || otherCell->numConnections < 3)) {
                adaptedDistance *= 2.0f;  // Reduce range of cell repulsion within creature by scaling distance
            }

            if (otherCell->fixed && adaptedDistance <= smoothingLength * 2 && cell->detached + otherCell->detached != 1) {
                auto index = atomicAdd(&numFixedCells, 1);
                if (index < MaxBarrierCellsForCollision) {
                    fixedCells[index] = otherCell;
                }
            }

            if (!otherCell->fixed && adaptedDistance <= smoothingLength * 2 && cell->detached + otherCell->detached != 1) {

                // Calc density
                atomicAdd_block(&density, calcKernel(adaptedDistance / smoothingLength) / (smoothingLength * smoothingLength));

                if (cell != otherCell) {

                    // Overlap correction
                    if (!cell->fixed && origDistance < cudaSimulationParameters.minCellDistance.value) {
                        atomicAdd_block(&cellPosDelta.x, posDelta.x * cudaSimulationParameters.minCellDistance.value / 5);
                        atomicAdd_block(&cellPosDelta.y, posDelta.y * cudaSimulationParameters.minCellDistance.value / 5);
                    }

                    auto velDelta = cell->vel - otherCell->vel;
                    bool isConnected = false;
                    for (int i = 0; i < cell->numConnections; ++i) {
                        auto const& connectedCell = cell->connections[i].cell;
                        if (connectedCell == otherCell) {
                            isConnected = true;
                        }
                    }
                    if (!isConnected) {

                        // Calc forces: for simplicity pressure = density
                        auto const& cellPressure = cell->density;            // Optimization: using the density from last time step
                        auto const& otherCellPressure = otherCell->density;  // Optimization: using the density from last time step
                        auto factor = cellPressure / (cell->density * cell->density) + otherCellPressure / (otherCell->density * otherCell->density);

                        if (adaptedDistance > NEAR_ZERO) {
                            float kernel_d = calcKernel_d(adaptedDistance / smoothingLength) / (smoothingLength * smoothingLength * smoothingLength);

                            auto F_pressureDelta = posDelta / (-adaptedDistance) * factor * kernel_d;
                            atomicAdd_block(&F_pressure.x, F_pressureDelta.x);
                            atomicAdd_block(&F_pressure.y, F_pressureDelta.y);

                            auto F_viscosityDelta = velDelta / otherCell->density * adaptedDistance * kernel_d / (adaptedDistance * adaptedDistance + 0.25f);
                            atomicAdd_block(&F_viscosity.x, F_viscosityDelta.x);
                            atomicAdd_block(&F_viscosity.y, F_viscosityDelta.y);
                        }
                    }

                    // Fusion
                    if (Math::length(velDelta) >= cellFusionVelocity && cell->numConnections < MAX_CELL_BONDS && otherCell->numConnections < MAX_CELL_BONDS
                        && (cell->sticky || otherCell->sticky) && cell->usableEnergy <= cellMaxBindingEnergy && otherCell->usableEnergy <= cellMaxBindingEnergy
                        && !cell->fixed && !otherCell->fixed) {
                        CellConnectionProcessor::scheduleAddConnectionPair(data, cell, otherCell);
                    }
                }
            }
            otherCell = otherCell->nextCell;
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            // Calculate fixed forces
            numFixedCells = min(MaxBarrierCellsForCollision, numFixedCells);
            if (numFixedCells > 0) {
                Cell* closestBarrierCell = nullptr;
                float closestBarrierCellDistance;
                for (int i = 0; i < numFixedCells; ++i) {
                    auto const& fixedCell = fixedCells[i];
                    auto distance = data.cellMap.getDistance(cell->pos, fixedCell->pos);
                    if (!closestBarrierCell || distance < closestBarrierCellDistance) {
                        closestBarrierCell = fixedCell;
                        closestBarrierCellDistance = distance;
                    }
                }

                float2 r{0, 0};
                if (closestBarrierCell->numConnections <= 1) {
                    r = data.cellMap.getCorrectedDirection(cell->pos - closestBarrierCell->pos);
                } else {
                    auto angleToCell = Math::angleOfVector(data.cellMap.getCorrectedDirection(cell->pos - closestBarrierCell->pos));
                    auto numConnections = closestBarrierCell->numConnections;
                    for (int i = 0; i < numConnections; ++i) {
                        auto otherCell1 = closestBarrierCell->connections[i].cell;
                        auto otherCell2 = closestBarrierCell->connections[(i + 1) % numConnections].cell;
                        auto angleToOtherCell1 = Math::angleOfVector(data.cellMap.getCorrectedDirection(otherCell1->pos - closestBarrierCell->pos));
                        auto angleToOtherCell2 = Math::angleOfVector(data.cellMap.getCorrectedDirection(otherCell2->pos - closestBarrierCell->pos));
                        if (Math::isAngleInBetween(angleToOtherCell1, angleToOtherCell2, angleToCell)) {
                            r = otherCell2->pos - otherCell1->pos;
                            Math::rotateQuarterCounterClockwise(r);
                            break;
                        }
                    }
                }
                auto vr = cell->vel - closestBarrierCell->vel;
                auto dot_vr_r = Math::dot(vr, r);

                if (dot_vr_r < 0) {
                    auto truncated_r_squared = max(0.05f, Math::lengthSquared(r));
                    auto truncated_distance = max(0.05f, closestBarrierCellDistance);
                    cell->shared1 += (vr - r * 2 * dot_vr_r / truncated_r_squared + closestBarrierCell->vel - cell->vel) / truncated_distance;
                }
            }

            cell->pos += cellPosDelta;
            cell->shared1 += F_pressure * cudaSimulationParameters.pressureStrength.value + F_viscosity * cudaSimulationParameters.viscosityStrength.value;
            cell->shared2.x = density;
        }
        __syncthreads();
    }
}

__inline__ __device__ void CellProcessor::calcCollisions_reconnectCells_correctOverlap(SimulationData& data)
{
    auto& cells = data.objects.cells;
    auto partition = calcSystemThreadPartition(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& cell = cells.at(index);
        data.cellMap.executeForEach(cell->pos, cudaSimulationParameters.maxCollisionDistance.value, cell->detached, [&](auto const& otherCell) {
            if (otherCell == cell) {
                return;
            }

            auto posDelta = cell->pos - otherCell->pos;
            data.cellMap.correctDirection(posDelta);

            auto distance = Math::length(posDelta);

            //overlap correction
            if (!cell->fixed) {
                if (distance < cudaSimulationParameters.minCellDistance.value) {
                    cell->pos += posDelta * cudaSimulationParameters.minCellDistance.value / 5;
                }
            }

            bool alreadyConnected = false;
            for (int i = 0; i < cell->numConnections; ++i) {
                auto const& connectedCell = cell->connections[i].cell;
                if (connectedCell == otherCell) {
                    alreadyConnected = true;
                    break;
                }
            }

            if (!alreadyConnected) {

                //collision algorithm
                auto velDelta = cell->vel - otherCell->vel;
                auto isApproaching = Math::dot(posDelta, velDelta) < 0;
                auto fixedFactor = cell->fixed ? 2 : 1;

                if (Math::length(cell->vel) > 0.5f && isApproaching) {
                    auto distanceSquared = distance * distance + 0.25f;
                    auto force = posDelta * Math::dot(velDelta, posDelta) / (-2 * distanceSquared) * fixedFactor;
                    atomicAdd(&cell->shared1.x, force.x);
                    atomicAdd(&cell->shared1.y, force.y);
                    atomicAdd(&otherCell->shared1.x, -force.x);
                    atomicAdd(&otherCell->shared1.y, -force.y);
                } else {
                    auto force = Math::getNormalized(posDelta) * (cudaSimulationParameters.maxCollisionDistance.value - Math::length(posDelta))
                        * cudaSimulationParameters.repulsionStrength.value * fixedFactor;
                    atomicAdd(&cell->shared1.x, force.x);
                    atomicAdd(&cell->shared1.y, force.y);
                    atomicAdd(&otherCell->shared1.x, -force.x);
                    atomicAdd(&otherCell->shared1.y, -force.y);
                }

                //fusion
                auto cellMaxBindingEnergy = ParameterCalculator::calcParameter(cudaSimulationParameters.cellMaxBindingEnergy, data, cell->pos);
                auto cellFusionVelocity = ParameterCalculator::calcParameter(cudaSimulationParameters.cellFusionVelocity, data, cell->pos);

                if (cell->numConnections < MAX_CELL_BONDS && otherCell->numConnections < MAX_CELL_BONDS && (cell->sticky || otherCell->sticky)
                    && Math::length(velDelta) >= cellFusionVelocity && isApproaching && cell->usableEnergy <= cellMaxBindingEnergy
                    && otherCell->usableEnergy <= cellMaxBindingEnergy && !cell->fixed && !otherCell->fixed) {
                    CellConnectionProcessor::scheduleAddConnectionPair(data, cell, otherCell);
                }
            }
        });
    }
}

__inline__ __device__ void CellProcessor::checkForces(SimulationData& data)
{
    auto& cells = data.objects.cells;
    auto const partition = calcSystemThreadPartition(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& cell = cells.at(index);
        cell->density = cell->shared2.x;
        if (cell->fixed) {
            continue;
        }

        if (Math::length(cell->shared1) > ParameterCalculator::calcParameter(cudaSimulationParameters.maxForce, data, cell->pos, cell->color)) {
            if (data.primaryNumberGen.random() < cudaSimulationParameters.maxForceDecayProbability) {
                CellConnectionProcessor::scheduleDeleteAllConnections(data, cell);
            }
        }
    }
}

__inline__ __device__ void CellProcessor::applyForces(SimulationData& data)
{
    auto& cells = data.objects.cells;
    auto const partition = calcSystemThreadPartition(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& cell = cells.at(index);
        if (cell->fixed) {
            continue;
        }

        cell->vel += cell->shared1;
        if (Math::length(cell->vel) > cudaSimulationParameters.maxVelocity.value) {
            cell->vel = Math::getNormalized(cell->vel) * cudaSimulationParameters.maxVelocity.value;
        }
        cell->shared1 = {0, 0};
    }
}

__inline__ __device__ void CellProcessor::calcConnectionForces(SimulationData& data, bool calcAngularForces)
{
    auto& cells = data.objects.cells;
    auto const partition = calcSystemThreadPartition(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& cell = cells.at(index);
        if (0 == cell->numConnections || cell->fixed) {
            continue;
        }
        float2 force{0, 0};
        float2 prevDisplacement = cell->connections[cell->numConnections - 1].cell->pos - cell->pos;
        data.cellMap.correctDirection(prevDisplacement);
        auto cellStiffnessSquared = cell->stiffness * cell->stiffness;

        auto numConnections = cell->numConnections;
        for (int i = 0; i < numConnections; ++i) {
            auto connectedCell = cell->connections[i].cell;
            auto connectedCellStiffnessSquared = connectedCell->stiffness * connectedCell->stiffness;

            auto displacement = connectedCell->pos - cell->pos;
            data.cellMap.correctDirection(displacement);

            auto actualDistance = Math::length(displacement);
            auto bondDistance = cell->connections[i].distance;
            auto deviation = actualDistance - bondDistance;
            force = force + Math::getNormalized(displacement) * deviation * (cellStiffnessSquared + connectedCellStiffnessSquared) / 6;

            if (calcAngularForces) {
                auto lastIndex = (i + numConnections - 1) % numConnections;
                auto lastConnectedCell = cell->connections[lastIndex].cell;

                auto referenceAngleFromPrevious = cell->connections[i].angleFromPrevious;

                auto r1 = prevDisplacement;
                auto r2 = displacement;
                Math::rotateQuarterClockwise(r1);
                Math::rotateQuarterCounterClockwise(r2);

                auto angle = Math::angleOfVector(displacement);
                auto prevAngle = Math::angleOfVector(prevDisplacement);
                auto theta = Math::getNormalizedAngle(angle - prevAngle, 0.0f);

                if (theta < referenceAngleFromPrevious) {
                    r1 *= -1.0f;
                    r2 *= -1.0f;
                }
                auto g = 5e-4f * abs(Math::getNormalizedAngle(theta - referenceAngleFromPrevious, -180.0f)) * cellStiffnessSquared;
                auto strength1 = g / max(Math::lengthSquared(r1), cudaSimulationParameters.minCellDistance.value);
                auto strength2 = g / max(Math::lengthSquared(r2), cudaSimulationParameters.minCellDistance.value);
                auto force2 = r1 * strength1;
                auto force1 = r2 * strength2;

                if (!connectedCell->fixed && !lastConnectedCell->fixed) {
                    atomicAdd(&connectedCell->shared1.x, force1.x);
                    atomicAdd(&connectedCell->shared1.y, force1.y);
                    atomicAdd(&lastConnectedCell->shared1.x, force2.x);
                    atomicAdd(&lastConnectedCell->shared1.y, force2.y);
                }
                force -= force1 + force2;
            }

            prevDisplacement = displacement;
        }
        atomicAdd(&cell->shared1.x, force.x);
        atomicAdd(&cell->shared1.y, force.y);
    }
}

__inline__ __device__ void CellProcessor::checkConnections(SimulationData& data)
{
    auto& cells = data.objects.cells;
    auto const partition = calcSystemThreadPartition(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& cell = cells.at(index);
        if (cell->fixed) {
            continue;
        }

        bool scheduleForDestruction = false;
        for (int i = 0; i < cell->numConnections; ++i) {
            auto connectedCell = cell->connections[i].cell;

            auto displacement = connectedCell->pos - cell->pos;
            data.cellMap.correctDirection(displacement);
            auto actualDistance = Math::length(displacement);
            if (actualDistance > cudaSimulationParameters.maxBindingDistance.value[cell->color]) {
                scheduleForDestruction = true;
            }
        }
        if (scheduleForDestruction) {
            CellConnectionProcessor::scheduleDeleteAllConnections(data, cell);
            for (int i = 0; i < cell->numConnections; ++i) {
                auto connectedCell = cell->connections[i].cell;
                connectedCell->cellState = CellState_Detaching;
            }
        }
    }
}

__inline__ __device__ void CellProcessor::verletPositionUpdate(SimulationData& data)
{
    auto& cells = data.objects.cells;
    auto const partition = calcSystemThreadPartition(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& cell = cells.at(index);
        if (cell->fixed) {
            cell->pos += cell->vel * cudaSimulationParameters.timestepSize.value;
            data.cellMap.correctPosition(cell->pos);
        } else {
            cell->pos += cell->vel * cudaSimulationParameters.timestepSize.value
                + cell->shared1 * cudaSimulationParameters.timestepSize.value * cudaSimulationParameters.timestepSize.value / 2;
            data.cellMap.correctPosition(cell->pos);
            cell->shared2 = cell->shared1;  //forces
            cell->shared1 = {0, 0};
        }
    }
}

__inline__ __device__ void CellProcessor::verletVelocityUpdate(SimulationData& data)
{
    auto& cells = data.objects.cells;
    auto const partition = calcSystemThreadPartition(cells.getNumOrigEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& cell = cells.at(index);
        if (cell->fixed) {
            continue;
        }
        auto acceleration = (cell->shared1 + cell->shared2) / 2;
        cell->vel += acceleration * cudaSimulationParameters.timestepSize.value;
    }
}

__inline__ __device__ void CellProcessor::aging(SimulationData& data)
{
    auto const partition = calcSystemThreadPartition(data.objects.cells.getNumEntries());
    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& cell = data.objects.cells.at(index);
        if (cell->fixed) {
            continue;
        }
        ++cell->age;


        if (cudaSimulationParameters.colorTransitionRulesToggle.value) {
            int transitionDuration;
            int targetColor;
            auto color = calcMod(cell->color, MAX_COLORS);
            auto index = ParameterCalculator::getFirstMatchingLayerOrBase(data, cell->pos, cudaSimulationParameters.colorTransitionRules);
            if (index == -1) {
                transitionDuration = cudaSimulationParameters.colorTransitionRules.baseValue[color].duration;
                targetColor = cudaSimulationParameters.colorTransitionRules.baseValue[color].targetColor;
            } else {
                transitionDuration = cudaSimulationParameters.colorTransitionRules.layerValues[index].value[color].duration;
                targetColor = cudaSimulationParameters.colorTransitionRules.layerValues[index].value[color].targetColor;
            }
            if (transitionDuration > 0 && cell->age > transitionDuration) {
                cell->color = targetColor;
                cell->age = 0;
            }
        }
        if (cell->cellState == CellState_Ready && cell->activationTime > 0) {
            --cell->activationTime;
        }
    }
}


__inline__ __device__ void CellProcessor::cellStateTransition_calcFutureState(SimulationData& data)
{
    auto& cells = data.objects.cells;
    auto partition = calcSystemThreadPartition(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& cell = cells.at(index);

        bool isSameCreatureNeighborDetaching = false;
        bool isOtherCreatureNeighborDetaching = false;
        bool isSameCreatureNeighborReviving = false;
        bool isNeighborActivating = false;
        //int activatingCellConnection = -1;
        for (int i = 0; i < cell->numConnections; ++i) {
            auto const& connectedCell = cell->connections[i].cell;
            if (cell->isSameCreature(connectedCell)) {
                auto connectedCellState = connectedCell->cellState;
                if (connectedCellState == CellState_Detaching) {
                    isSameCreatureNeighborDetaching = true;
                } else if (connectedCellState == CellState_Reviving) {
                    isSameCreatureNeighborReviving = true;
                } else if (connectedCellState == CellState_Activating) {
                    if (connectedCell->connections[0].cell == cell) {
                        isNeighborActivating = true;
                        //activatingCellConnection = i;
                    }
                }
            } else {
                if (connectedCell->cellState == CellState_Detaching) {
                    isOtherCreatureNeighborDetaching = true;
                }
            }
        }

        auto origCellState = cell->cellState;
        auto cellState = origCellState;

        if (cell->fixed) {
            cellState = CellState_Ready;
        } else {
            if (origCellState == CellState_Activating) {
                cellState = CellState_Ready;
                if (cudaSimulationParameters.cellAgeLimiterToggle.value && cudaSimulationParameters.resetCellAgeAfterActivation.value) {
                    atomicExch(&cell->age, 0);
                }
            } else if (origCellState == CellState_Reviving) {
                cellState = CellState_Ready;
            } else if (origCellState == CellState_Constructing) {
                if (isNeighborActivating) {
                    cellState = CellState_Activating;
                }
                if (isOtherCreatureNeighborDetaching && cudaSimulationParameters.cellDeathConsequences.value != CellDeathConsequences_None) {
                    cellState = CellState_Detaching;
                }
            } else if (origCellState == CellState_Detaching) {
                if (isSameCreatureNeighborReviving && cudaSimulationParameters.cellDeathConsequences.value == CellDeathConsequences_DetachedPartsDie) {
                    cellState = CellState_Reviving;
                }
                if (cudaSimulationParameters.cellDeathConsequences.value == CellDeathConsequences_None) {
                    cellState = CellState_Ready;
                }
            } else if (origCellState == CellState_Ready) {
                if (isSameCreatureNeighborDetaching && cudaSimulationParameters.cellDeathConsequences.value != CellDeathConsequences_None) {
                    if (cudaSimulationParameters.cellDeathConsequences.value == CellDeathConsequences_DetachedPartsDie && cell->creature != nullptr
                        && cell->headCell) {
                        cellState = CellState_Reviving;
                    } else {
                        cellState = CellState_Detaching;
                    }
                }
            }
            if (cell->cellType == CellType_Free || cell->cellType == CellType_Structure) {
                cellState = origCellState;
            }
        }
        cell->tempValue.as_uint32_float.uint32Part = cellState;
    }
}

__inline__ __device__ void CellProcessor::cellStateTransition_applyNextState(SimulationData& data)
{
    auto& cells = data.objects.cells;
    auto partition = calcSystemThreadPartition(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& cell = cells.at(index);
        cell->cellState = cell->tempValue.as_uint32_float.uint32Part;
        cell->tempValue.as_uint32_float.uint32Part = 0;
    }
}

__inline__ __device__ void CellProcessor::frontAngleUpdate_calcFutureValue(SimulationData& data)
{
    auto& cells = data.objects.cells;
    auto partition = calcSystemThreadPartition(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& cell = cells.at(index);
        if (cell->creature == nullptr) {
            continue;
        }
        if (cell->frontAngleId != cell->creature->frontAngleId) {
            if (!cell->headCell) {
                auto update = false;
                for (int i = 0, j = cell->numConnections; i < j; ++i) {
                    auto const& otherCell = cell->connections[i].cell;
                    if (!cell->isSameCreature(otherCell)) {
                        continue;
                    }
                    if (otherCell->frontAngleId == cell->creature->frontAngleId) {
                        auto frontAngle_otherCell_cell =
                            Math::getNormalizedAngle(otherCell->frontAngle + getInitialAngelSpan(otherCell, cell, otherCell->connections[0].cell), -180.0f);
                        auto frontAngle_cell_otherCell = Math::getNormalizedAngle(frontAngle_otherCell_cell - 180.0f, -180.0f);
                        auto frontAngle_cell_connection0 = Math::getNormalizedAngle(frontAngle_cell_otherCell + getInitialAngelSpan(cell, 0, i), -180.0f);
                        cell->tempValue.as_uint32_float.floatPart = frontAngle_cell_connection0;

                        update = true;
                        break;
                    }
                }
                if (!update) {
                    cell->tempValue.as_uint32_float.floatPart = FrontAngleId_NoUpdate;
                }
            }
        }
    }
}

__inline__ __device__ void CellProcessor::frontAngleUpdate_applyFutureValue(SimulationData& data)
{
    auto& cells = data.objects.cells;
    auto partition = calcSystemThreadPartition(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& cell = cells.at(index);
        if (cell->creature == nullptr) {
            continue;
        }
        if (cell->frontAngleId != cell->creature->frontAngleId) {
            if (cell->headCell) {
                cell->frontAngleId = cell->creature->frontAngleId;
                cell->frontAngle = cell->creature->genome->frontAngle;
            } else {
                if (cell->tempValue.as_uint32_float.floatPart != FrontAngleId_NoUpdate) {
                    cell->frontAngleId = cell->creature->frontAngleId;
                    cell->frontAngle = cell->tempValue.as_uint32_float.floatPart;
                }
                cell->tempValue.as_uint32_float.floatPart = 0;
            }
        }
    }
}

__inline__ __device__ void CellProcessor::applyInnerFriction(SimulationData& data)
{
    auto& cells = data.objects.cells;
    auto const partition = calcSystemThreadPartition(cells.getNumEntries());

    auto const innerFriction = cudaSimulationParameters.innerFriction.value;
    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& cell = cells.at(index);
        if (cell->fixed) {
            continue;
        }
        for (int index = 0; index < cell->numConnections; ++index) {
            auto connectingCell = cell->connections[index].cell;
            if (connectingCell->fixed) {
                continue;
            }
            auto posDelta = cell->pos - connectingCell->pos;
            auto distance = Math::length(posDelta);
            if (distance > NEAR_ZERO) {
                auto direction = posDelta / distance;
                auto velDelta = cell->vel - connectingCell->vel;
                auto velDelta_part = Math::dot(velDelta, direction);

                auto delta = direction * innerFriction * velDelta_part;
                atomicAdd(&cell->vel.x, -delta.x * 0.5f);
                atomicAdd(&cell->vel.y, -delta.y * 0.5f);
                atomicAdd(&connectingCell->vel.x, delta.x * 0.5f);
                atomicAdd(&connectingCell->vel.y, delta.y * 0.5f);
            }
        }
    }
}

__inline__ __device__ void CellProcessor::applyFriction(SimulationData& data)
{
    auto& cells = data.objects.cells;
    auto const partition = calcSystemThreadPartition(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& cell = cells.at(index);
        if (cell->fixed) {
            continue;
        }

        auto friction = ParameterCalculator::calcParameter(cudaSimulationParameters.friction, data, cell->pos);
        cell->vel = cell->vel * (1.0f - friction);
    }
}

__inline__ __device__ void CellProcessor::radiation(SimulationData& data)
{
    auto& cells = data.objects.cells;

    auto partition = calcSystemThreadPartition(cells.getNumEntries());
    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& cell = cells.at(index);
        if (cell->fixed) {
            continue;
        }
        if (cell->cellType == CellType_Structure) {
            continue;
        }
        if (data.primaryNumberGen.random() < cudaSimulationParameters.radiationProbability) {

            auto radiation1 = 0.0f;
            auto radiation2 = 0.0f;
            auto const& cellEnergy = cell->usableEnergy;
            auto const& cellRawEnergy = cell->rawEnergy;
            if (cell->usableEnergy > cudaSimulationParameters.radiationType2_energyThreshold.value[cell->color]) {
                radiation1 += cudaSimulationParameters.radiationType2_strength.value[cell->color];
            }
            if (cell->rawEnergy > cudaSimulationParameters.radiationType2_energyThreshold.value[cell->color]) {
                radiation2 += cudaSimulationParameters.radiationType2_strength.value[cell->color];
            }
            if (cell->age > cudaSimulationParameters.radiationType1_minimumAge.value[cell->color]) {
                radiation1 += ParameterCalculator::calcParameter(cudaSimulationParameters.radiationType1_strength, data, cell->pos, cell->color);
                radiation2 += ParameterCalculator::calcParameter(cudaSimulationParameters.radiationType1_strength, data, cell->pos, cell->color);
            }
            radiation1 *= cellEnergy;
            radiation2 *= cellRawEnergy;

            radiation1 = max(min(radiation1 / cudaSimulationParameters.radiationProbability * data.primaryNumberGen.random() * 2, cellEnergy - 1), 0.0f);
            radiation2 = max(min(radiation2 / cudaSimulationParameters.radiationProbability * data.primaryNumberGen.random() * 2, cellRawEnergy - 1), 0.0f);

            if (radiation1 > 0 || radiation2 > 0) {
                float2 particleVel = cell->vel * cudaSimulationParameters.radiationVelocityMultiplier
                    + Math::unitVectorOfAngle(data.primaryNumberGen.random() * 360) * cudaSimulationParameters.radiationVelocityPerturbation;
                float2 particlePos = cell->pos + Math::getNormalized(particleVel) * 1.5f
                    - particleVel;  // minus particleVel because particle will still be moved in current time step
                data.cellMap.correctPosition(particlePos);

                EnergyParticleProcessor::createEnergyParticle(data, particlePos, particleVel, cell->color, radiation1 + radiation2);

                cell->usableEnergy -= radiation1;
                cell->rawEnergy -= radiation2;
            }
        }
    }
}

__inline__ __device__ void CellProcessor::decay(SimulationData& data)
{
    auto& cells = data.objects.cells;
    auto partition = calcSystemThreadPartition(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& cell = cells.at(index);
        if (cell->fixed) {
            continue;
        }
        auto cellMaxBindingEnergy = ParameterCalculator::calcParameter(cudaSimulationParameters.cellMaxBindingEnergy, data, cell->pos);
        if (cell->usableEnergy > cellMaxBindingEnergy) {
            CellConnectionProcessor::scheduleDeleteAllConnections(data, cell);
        }

        auto cellMinEnergy = ParameterCalculator::calcParameter(cudaSimulationParameters.minCellEnergy, data, cell->pos, cell->color);

        if (cell->cellState == CellState_Dying || cell->cellState == CellState_Detaching) {
            auto cellDeathProbability = ParameterCalculator::calcParameter(cudaSimulationParameters.cellDeathProbability, data, cell->pos, cell->color);
            if (data.primaryNumberGen.random() <= cellDeathProbability) {
                CellConnectionProcessor::scheduleDeleteCell(data, index);
            }
        }

        bool cellDestruction = false;
        if (cell->usableEnergy < cellMinEnergy) {
            cellDestruction = true;
        }

        auto cellMaxAge = cudaSimulationParameters.maxCellAge.value[cell->color];
        if (cudaSimulationParameters.cellAgeLimiterToggle.value && cell->cellType != CellType_Free && cell->cellType != CellType_Structure
            && cell->cellTriggered == CellTriggered_No && cell->cellState == CellState_Ready && cell->activationTime == 0) {
            bool adjacentCellsUsed = false;
            for (int i = 0; i < cell->numConnections; ++i) {
                if (cell->connections[i].cell->cellTriggered == CellTriggered_Yes) {
                    adjacentCellsUsed = true;
                    break;
                }
            }
            if (!adjacentCellsUsed) {
                auto cellInactiveMaxAge = ParameterCalculator::calcParameter(cudaSimulationParameters.maxAgeForInactiveCells, data, cell->pos, cell->color);

                cellMaxAge = toInt(cellInactiveMaxAge);
            }
        }
        if (cudaSimulationParameters.cellAgeLimiterToggle.value && cell->cellType == CellType_Free) {
            cellMaxAge = cudaSimulationParameters.freeCellMaxAge.value[cell->color];
        }
        if (cellMaxAge > 0 && cell->age > cellMaxAge) {
            cellDestruction = true;
        }

        if (cellDestruction) {
            auto orig = atomicExch(&cell->cellState, CellState_Dying);
            if (orig != CellState_Dying) {
                for (int i = 0; i < cell->numConnections; ++i) {
                    auto const& connectedCell = cell->connections[i].cell;
                    auto origConnected = atomicExch(&connectedCell->cellState, CellState_Detaching);
                    if (origConnected == CellState_Dying) {
                        atomicExch(&connectedCell->cellState, CellState_Dying);
                    }
                }
            }
        }
    }
}

__inline__ __device__ void CellProcessor::resetDensity(SimulationData& data)
{
    auto& cells = data.objects.cells;
    auto partition = calcSystemThreadPartition(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& cell = cells.at(index);

        cell->density = 1.0f;
    }
}

__inline__ __device__ void CellProcessor::updateCellEvents(SimulationData& data)
{
    auto& cells = data.objects.cells;
    auto partition = calcSystemThreadPartition(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& cell = cells.at(index);
        if (cell->eventCounter > 0) {
            --cell->eventCounter;
        }
    }
}

__inline__ __device__ void CellProcessor::performEnergyFlow(SimulationData& data)
{
    auto& cells = data.objects.cells;
    auto partition = calcSystemThreadPartition(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& cell = cells.at(index);
        if (cell->numConnections == 0) {
            continue;
        }
        //if (cell->cellState == CellState_Constructing || cell->cellState == CellState_Activating) {
        //    continue;
        //}
        auto i = data.timestep % cell->numConnections;
        auto& connectedCell = cell->connections[i].cell;

        // Flow of usable energy
        {
            auto cellMinEnergy = ParameterCalculator::calcParameter(cudaSimulationParameters.minCellEnergy, data, cell->pos, cell->color);
            auto cellNormalEnergy = cudaSimulationParameters.normalCellEnergy.value[cell->color];

            auto needsEnergy = [](Cell* cell) {
                return (cell->cellState == CellState_Ready || cell->cellState == CellState_Detaching || cell->cellState == CellState_Reviving)
                    && cell->cellType == CellType_Constructor && cell->creature
                    && !ConstructorHelper::isFinished(cell->cellTypeData.constructor, *cell->creature->genome);
            };
            auto lowEnergy = [&](Cell* cell) { return cell->usableEnergy < cellNormalEnergy; };

            auto cellNeedsEnergy = needsEnergy(cell);
            auto connectedCellNeedsEnergy = needsEnergy(connectedCell);
            auto connectedCellLowEnergy = lowEnergy(cell);

            auto flow = 0.0f;
            if (connectedCellLowEnergy) {
                if (cell->usableEnergy > connectedCell->usableEnergy) {
                    flow = (cell->usableEnergy - connectedCell->usableEnergy) / 2;
                }
            } else {
                if ((cellNeedsEnergy && connectedCellNeedsEnergy) || (!cellNeedsEnergy && !connectedCellNeedsEnergy)) {
                    if (cell->usableEnergy > connectedCell->usableEnergy) {
                        flow = (cell->usableEnergy - connectedCell->usableEnergy) / 2;
                    }
                }
                if (!cellNeedsEnergy && connectedCellNeedsEnergy) {
                    if (cell->usableEnergy > cellNormalEnergy) {
                        flow = cell->usableEnergy - cellNormalEnergy;
                    }
                }
            }

            if (flow > 0) {
                flow = min(2.0f, flow);
                auto orig = atomicAdd(&cell->usableEnergy, -flow);
                if (orig < cellMinEnergy) {
                    atomicAdd(&cell->usableEnergy, flow);
                } else {
                    atomicAdd(&connectedCell->usableEnergy, flow);
                }
            }
        }

        // Flow of raw energy
        {
            if ((cell->cellState == CellState_Ready || cell->cellState == CellState_Detaching || cell->cellState == CellState_Reviving)
                && (connectedCell->cellState == CellState_Ready || connectedCell->cellState == CellState_Detaching
                    || connectedCell->cellState == CellState_Reviving)
                && connectedCell->rawEnergy < SimulationParameters::maxRawEnergyThresholdForConduction) {

                auto flow = 0.0f;
                auto maxFlow = 0.0f;
                if (connectedCell->cellType == CellType_Digestor) {
                    maxFlow += connectedCell->cellTypeData.digestor.rawEnergyConductivity
                        * cudaSimulationParameters.maxRawEnergyConductivity.value[connectedCell->color];
                    if (cell->cellType == CellType_Digestor) {
                        maxFlow += cell->cellTypeData.digestor.rawEnergyConductivity * cudaSimulationParameters.maxRawEnergyConductivity.value[cell->color];

                        auto cellConversionRate = 1.0f - cell->cellTypeData.digestor.rawEnergyConductivity;
                        auto connectedCellConversionRate = 1.0f - connectedCell->cellTypeData.digestor.rawEnergyConductivity;
                        if (cellConversionRate == 0 && connectedCellConversionRate == 0) {
                            cellConversionRate = 1;
                            connectedCellConversionRate = 1;
                        }

                        auto targetEnergy =
                            (cell->rawEnergy + connectedCell->rawEnergy) * cellConversionRate / (cellConversionRate + connectedCellConversionRate);
                        flow = max(cell->rawEnergy - targetEnergy, 0.0f);
                    } else {
                        flow = cell->rawEnergy;
                    }
                }

                flow = min(maxFlow, flow);
                auto orig = atomicAdd(&cell->rawEnergy, -flow);
                if (orig < 0) {
                    atomicAdd(&cell->rawEnergy, flow);
                } else {
                    atomicAdd(&connectedCell->rawEnergy, flow);
                }
            }
        }
    }
}

__device__ __inline__ float CellProcessor::getInitialAngelSpan(Cell* cell, int connectionIndex1, int connectionIndex2)
{
    if ((connectionIndex1 - connectionIndex2 + cell->numConnections) % cell->numConnections == 0) {
        return 0;
    }
    auto result = 0.0f;
    for (int i = connectionIndex1 + 1; i < connectionIndex1 + cell->numConnections; i++) {
        auto index = i % cell->numConnections;
        result += MuscleProcessor::getInitialAngleFromPrevious(cell, index);
        if (index == connectionIndex2) {
            break;
        }
    }
    return Math::getNormalizedAngle(result, -180.0f);
}

__device__ __inline__ float CellProcessor::getInitialAngelSpan(Cell* cell, Cell* connectedCell1, Cell* connectedCell2)
{
    auto connectionIndex1 = -1;
    auto connectionIndex2 = -1;
    for (int i = 0; i < cell->numConnections; i++) {
        if (cell->connections[i].cell == connectedCell1) {
            connectionIndex1 = i;
        }
        if (cell->connections[i].cell == connectedCell2) {
            connectionIndex2 = i;
        }
    }
    if (connectionIndex1 == -1 || connectionIndex2 == -1) {
        return 0;
    }
    return getInitialAngelSpan(cell, connectionIndex1, connectionIndex2);
}
