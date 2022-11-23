#pragma once

#include "EngineInterface/Enums.h"

#include "Cell.cuh"
#include "CellFunctionProcessor.cuh"
#include "ConstantMemory.cuh"
#include "SimulationData.cuh"
#include "SpotCalculator.cuh"
#include "SimulationResult.cuh"
#include "ObjectFactory.cuh"

class AttackerProcessor
{
public:
    __inline__ __device__ static void process(SimulationData& data, SimulationResult& result);

private:
    __inline__ __device__ static void processCell(SimulationData& data, SimulationResult& result, Cell* cell);
    __inline__ __device__ static void radiate(SimulationData& data, Cell* cell);
    __inline__ __device__ static void distributeEnergy(SimulationData& data, Cell* cell, float energyDelta);

    __inline__ __device__ static float calcOpenAngle(Cell* cell, float2 direction);
    __inline__ __device__ static bool isConnectedConnected(Cell* cell, Cell* otherCell);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

namespace
{
    float constexpr AttackerActivityThreshold = 0.25f;
    float constexpr AttackRadius = 1.6f;
    float constexpr OutputPoisoned = -1;
    float constexpr OutputNothingFound = 0;
    float constexpr OutputSuccess = 1;
    float constexpr EnergyDistributionRadius = 3.0f;
}

__device__ __inline__ void AttackerProcessor::process(SimulationData& data, SimulationResult& result)
{
    auto& operations = data.cellFunctionOperations[Enums::CellFunction_Attacker];
    auto partition = calcAllThreadsPartition(operations.getNumEntries());
    for (int i = partition.startIndex; i <= partition.endIndex; ++i) {
        auto const& cell = operations.at(i).cell;
        processCell(data, result, cell);
    }
}

__device__ __inline__ void AttackerProcessor::processCell(SimulationData& data, SimulationResult& result, Cell* cell)
{
    auto activity = CellFunctionProcessor::calcInputActivity(cell);
    if (abs(activity.channels[0]) < AttackerActivityThreshold) {
        return;
    }
    if (!cell->tryLock()) {
        return;
    }
    float energyDelta = 0;

    Cell* otherCells[18];
    int numOtherCells;
    data.cellMap.get(otherCells, 18, numOtherCells, cell->absPos, AttackRadius);
    for (int i = 0; i < numOtherCells; ++i) {
        Cell* otherCell = otherCells[i];
        if (!otherCell->tryLock()) {
            continue;
        }
        if (!isConnectedConnected(cell, otherCell) && !otherCell->barrier) {
            auto energyToTransfer = otherCell->energy * cudaSimulationParameters.cellFunctionWeaponStrength + 1.0f;
            auto cellFunctionWeaponGeometryDeviationExponent =
                SpotCalculator::calcParameter(&SimulationParametersSpotValues::cellFunctionWeaponGeometryDeviationExponent, data, cell->absPos);

            if (abs(cellFunctionWeaponGeometryDeviationExponent) > 0) {
                auto d = otherCell->absPos - cell->absPos;
                auto angle1 = calcOpenAngle(cell, d);
                auto angle2 = calcOpenAngle(otherCell, d * (-1));
                auto deviation = 1.0f - abs(360.0f - (angle1 + angle2)) / 360.0f;  //1 = no deviation, 0 = max deviation
                energyToTransfer *= powf(max(0.0f, min(1.0f, deviation)), cellFunctionWeaponGeometryDeviationExponent);
            }

            auto color = calcMod(cell->color, MAX_COLORS);
            auto otherColor = calcMod(otherCell->color, MAX_COLORS);
            energyToTransfer *= SpotCalculator::calcColorMatrix(color, otherColor, data, cell->absPos);

            if (energyToTransfer >= 0) {
                if (otherCell->energy > energyToTransfer) {
                    otherCell->energy -= energyToTransfer;
                    energyDelta += energyToTransfer;
                }
            } else {
                otherCell->energy -= energyToTransfer;
                energyDelta += energyToTransfer;
        }

        }
        otherCell->releaseLock();
    }

    if (energyDelta >= 0) {
        distributeEnergy(data, cell, energyDelta);
    } else {
        auto cellMinEnergy = SpotCalculator::calcParameter(&SimulationParametersSpotValues::cellMinEnergy, data, cell->absPos);
        if (cell->energy > cellMinEnergy - energyDelta) {
            cell->energy += energyDelta;
        }
    }

    radiate(data, cell);

    //output
    if (energyDelta == 0) {
        cell->activity.channels[0] = OutputNothingFound;
        result.incFailedAttack();
    }
    if (energyDelta > 0) {
        cell->activity.channels[0] = OutputSuccess;
        result.incSuccessfulAttack();
    }
    if (energyDelta < 0) {
        cell->activity.channels[0] = OutputPoisoned;
    }

    cell->releaseLock();

    CellFunctionProcessor::setActivity(cell, activity);
}

__device__ __inline__ void AttackerProcessor::radiate(SimulationData& data, Cell* cell)
{
    auto cellFunctionWeaponEnergyCost = SpotCalculator::calcParameter(&SimulationParametersSpotValues::cellFunctionWeaponEnergyCost, data, cell->absPos);
    if (cellFunctionWeaponEnergyCost > 0) {
        auto const cellEnergy = cell->energy;
        auto& pos = cell->absPos;
        float2 particleVel = (cell->vel * cudaSimulationParameters.radiationVelocityMultiplier)
            + float2{
                (data.numberGen1.random() - 0.5f) * cudaSimulationParameters.radiationVelocityPerturbation,
                (data.numberGen1.random() - 0.5f) * cudaSimulationParameters.radiationVelocityPerturbation};
        float2 particlePos = pos + Math::normalized(particleVel) * 1.5f;
        data.cellMap.correctPosition(particlePos);

        particlePos = particlePos - particleVel;  //because particle will still be moved in current time step
        auto const radiationEnergy = min(cellEnergy, cellFunctionWeaponEnergyCost);
        cell->energy -= radiationEnergy;
        ObjectFactory factory;
        factory.init(&data);
        factory.createParticle(radiationEnergy, particlePos, particleVel, {cell->color});
    }
}

__device__ __inline__ void AttackerProcessor::distributeEnergy(SimulationData& data, Cell* cell, float energyDelta)
{
    Cell* receiverCells[10];
    int numReceivers;
    data.cellMap.getConstructorsAndTransmitters(receiverCells, 10, numReceivers, cell->absPos, EnergyDistributionRadius);
    float energyPerReceiver = numReceivers > 0 ? energyDelta / numReceivers : 0;

    for (int i = 0; i < numReceivers; ++i) {
        auto receiverCell = receiverCells[i];
        if (receiverCell->tryLock()) {
            receiverCell->energy += energyPerReceiver;
            energyDelta -= energyPerReceiver;
            receiverCell->releaseLock();
        }
    }
    cell->energy += energyDelta;
}

__inline__ __device__ float AttackerProcessor::calcOpenAngle(Cell* cell, float2 direction)
{
    if (0 == cell->numConnections) {
        return 0.0f;
    }
    if (1 == cell->numConnections) {
        return 365.0f;
    }

    auto refAngle = Math::angleOfVector(direction);

    float largerAngle = Math::angleOfVector(cell->connections[0].cell->absPos - cell->absPos);
    float smallerAngle = largerAngle;

    for (int i = 1; i < cell->numConnections; ++i) {
        auto otherCell = cell->connections[i].cell;
        auto angle = Math::angleOfVector(otherCell->absPos - cell->absPos);
        if (largerAngle >= refAngle) {
            if (largerAngle > angle && angle >= refAngle) {
                largerAngle = angle;
            }
        } else {
            if (largerAngle > angle || angle >= refAngle) {
                largerAngle = angle;
            }
        }

        if (smallerAngle <= refAngle) {
            if (smallerAngle < angle && angle <= refAngle) {
                smallerAngle = angle;
            }
        } else {
            if (smallerAngle < angle || angle <= refAngle) {
                smallerAngle = angle;
            }
        }
    }
    return Math::subtractAngle(largerAngle, smallerAngle);
}

__inline__ __device__ bool AttackerProcessor::isConnectedConnected(Cell* cell, Cell* otherCell)
{
    if (cell == otherCell) {
        return true;
    }
    bool result = false;
    for (int i = 0; i < otherCell->numConnections; ++i) {
        auto const& connectedCell = otherCell->connections[i].cell;
        if (connectedCell == cell) {
            result = true;
            break;
        }

        for (int j = 0; j < connectedCell->numConnections; ++j) {
            auto const& connectedConnectedCell = connectedCell->connections[j].cell;
            if (connectedConnectedCell == cell) {
                result = true;
                break;
            }
        }
        if (result) {
            return true;
        }
    }
    return result;
}
