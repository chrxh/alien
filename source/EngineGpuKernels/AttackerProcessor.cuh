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
    __inline__ __device__ static bool isHomogene(Cell* cell);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

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
    int inputExecutionOrderNumber;
    auto activity = CellFunctionProcessor::calcInputActivity(cell, inputExecutionOrderNumber);
    if (abs(activity.channels[0]) < cudaSimulationParameters.cellFunctionAttackerActivityThreshold) {
        return;
    }
    if (!cell->tryLock()) {
        return;
    }
    float energyDelta = 0;
    auto cellMinEnergy = SpotCalculator::calcParameter(&SimulationParametersSpotValues::cellMinEnergy, data, cell->absPos);

    Cell* otherCells[18];
    int numOtherCells;
    data.cellMap.get(otherCells, 18, numOtherCells, cell->absPos, cudaSimulationParameters.cellFunctionAttackerRadius);
    for (int i = 0; i < numOtherCells; ++i) {
        Cell* otherCell = otherCells[i];
        if (!otherCell->tryLock()) {
            continue;
        }
        if (!isConnectedConnected(cell, otherCell) && !otherCell->barrier) {
            auto energyToTransfer = otherCell->energy * cudaSimulationParameters.cellFunctionAttackerStrength + 1.0f;

            if (!isHomogene(otherCell)) {
                energyToTransfer *= cudaSimulationParameters.cellFunctionAttackerInhomogeneityBonus;
            }
            auto cellFunctionWeaponGeometryDeviationExponent =
                SpotCalculator::calcParameter(&SimulationParametersSpotValues::cellFunctionAttackerGeometryDeviationExponent, data, cell->absPos);

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
                if (cell->energy >= cellMinEnergy - (energyDelta + energyToTransfer)) {
                    otherCell->energy -= energyToTransfer;
                    energyDelta += energyToTransfer;
                }
        }

        }
        otherCell->releaseLock();
    }

    if (energyDelta > NEAR_ZERO) {
        distributeEnergy(data, cell, energyDelta);
    } else {
        cell->energy += energyDelta;
    }

    radiate(data, cell);

    //output
    if (energyDelta > NEAR_ZERO) {
        activity.channels[0] = cudaSimulationParameters.cellFunctionAttackerOutputSuccess;
        result.incSuccessfulAttack();
    } else if (energyDelta < -NEAR_ZERO) {
        activity.channels[0] = cudaSimulationParameters.cellFunctionAttackerOutputPoisoned;
        result.incFailedAttack();
    } else {
        activity.channels[0] = cudaSimulationParameters.cellFunctionAttackerOutputNothingFound;
        result.incFailedAttack();
    }

    cell->releaseLock();

    CellFunctionProcessor::setActivity(cell, activity);
}

__device__ __inline__ void AttackerProcessor::radiate(SimulationData& data, Cell* cell)
{
    auto cellFunctionWeaponEnergyCost = SpotCalculator::calcParameter(&SimulationParametersSpotValues::cellFunctionAttackerEnergyCost, data, cell->absPos);
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
    if (cell->energy > cudaSimulationParameters.cellNormalEnergy) {
        energyDelta += cell->energy - cudaSimulationParameters.cellNormalEnergy;
        cell->energy = cudaSimulationParameters.cellNormalEnergy;
    }

    if (cell->cellFunctionData.attacker.mode == Enums::EnergyDistributionMode_ConnectedCells) {
        int numReceivers = cell->numConnections;
        for (int i = 0; i < cell->numConnections; ++i) {
            numReceivers += cell->connections[i].cell->numConnections;
        }
        float energyPerReceiver = energyDelta / (numReceivers + 1);

        for (int i = 0; i < cell->numConnections; ++i) {
            auto connectedCell = cell->connections[i].cell;
            if (connectedCell->tryLock()) {
                connectedCell->energy += energyPerReceiver;
                energyDelta -= energyPerReceiver;
                connectedCell->releaseLock();
            }
            for (int i = 0; i < connectedCell->numConnections; ++i) {
                auto connectedConnectedCell = connectedCell->connections[i].cell;
                if (connectedConnectedCell->tryLock()) {
                    connectedConnectedCell->energy += energyPerReceiver;
                    energyDelta -= energyPerReceiver;
                    connectedConnectedCell->releaseLock();
                }
            }
        }
    }

    if (cell->cellFunctionData.attacker.mode == Enums::EnergyDistributionMode_TransmittersAndConstructors) {
        Cell* receiverCells[10];
        int numReceivers;
        data.cellMap.getCellsWithGivenFunction(
            receiverCells,
            10,
            numReceivers,
            cell->absPos,
            cudaSimulationParameters.cellFunctionAttackerEnergyDistributionRadius,
            Enums::CellFunction_Constructor);
        if (numReceivers == 0) {
            data.cellMap.getCellsWithGivenFunction(
                receiverCells,
                10,
                numReceivers,
                cell->absPos,
                cudaSimulationParameters.cellFunctionAttackerEnergyDistributionRadius,
                Enums::CellFunction_Transmitter);
        }
        float energyPerReceiver = energyDelta / (numReceivers + 1);

        for (int i = 0; i < numReceivers; ++i) {
            auto receiverCell = receiverCells[i];
            if (receiverCell->tryLock()) {
                receiverCell->energy += energyPerReceiver;
                energyDelta -= energyPerReceiver;
                receiverCell->releaseLock();
            }
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

__inline__ __device__ bool AttackerProcessor::isHomogene(Cell* cell)
{
    int color = cell->color;
    for (int i = 0; i < cell->numConnections; ++i) {
        auto otherCell = cell->connections[i].cell;
        if (color != otherCell->color) {
            return false;
        }
        for (int j = 0; j < otherCell->numConnections; ++j) {
            auto otherOtherCell = otherCell->connections[j].cell;
            if (color != otherOtherCell->color ) {
                return false;
            }
        }
    }
    return true;
}