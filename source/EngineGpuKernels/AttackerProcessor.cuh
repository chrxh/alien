#pragma once


#include "EngineInterface/CellFunctionConstants.h"

#include "Cell.cuh"
#include "CellFunctionProcessor.cuh"
#include "ConstantMemory.cuh"
#include "SimulationData.cuh"
#include "SpotCalculator.cuh"
#include "SimulationStatistics.cuh"
#include "ObjectFactory.cuh"
#include "ParticleProcessor.cuh"

class AttackerProcessor
{
public:
    __inline__ __device__ static void process(SimulationData& data, SimulationStatistics& result);

private:
    __inline__ __device__ static void processCell(SimulationData& data, SimulationStatistics& statistics, Cell* cell);
    __inline__ __device__ static void radiate(SimulationData& data, Cell* cell);
    __inline__ __device__ static void distributeEnergy(SimulationData& data, Cell* cell, float energyDelta);

    __inline__ __device__ static float calcOpenAngle(Cell* cell, float2 direction);
    __inline__ __device__ static bool isConnectedConnected(Cell* cell, Cell* otherCell);
    __inline__ __device__ static int countAndTrackDefenderCells(SimulationStatistics& statistics, Cell* cell);
    __inline__ __device__ static bool isHomogene(Cell* cell);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__device__ __inline__ void AttackerProcessor::process(SimulationData& data, SimulationStatistics& result)
{
    auto& operations = data.cellFunctionOperations[CellFunction_Attacker];
    auto partition = calcAllThreadsPartition(operations.getNumEntries());
    for (int i = partition.startIndex; i <= partition.endIndex; ++i) {
        processCell(data, result, operations.at(i).cell);
    }
}

__device__ __inline__ void AttackerProcessor::processCell(SimulationData& data, SimulationStatistics& statistics, Cell* cell)
{
    auto activity = CellFunctionProcessor::calcInputActivity(cell);
    if (abs(activity.channels[0]) < cudaSimulationParameters.cellFunctionAttackerActivityThreshold) {
        return;
    }
    float energyDelta = 0;
    auto cellMinEnergy =
        SpotCalculator::calcParameter(&SimulationParametersSpotValues::cellMinEnergy, &SimulationParametersSpotActivatedValues::cellMinEnergy, data, cell->absPos, cell->color);

    Cell* someOtherCell = nullptr;
    data.cellMap.executeForEach(cell->absPos, cudaSimulationParameters.cellFunctionAttackerRadius[cell->color], cell->detached, [&](auto const& otherCell) {
        if (cell->creatureId != 0 && otherCell->creatureId == cell->creatureId) {
            return;
        }
        if (cell->creatureId == 0 && isConnectedConnected(cell, otherCell)) {
            return;
        }
        if (!otherCell->barrier /*&& otherCell->livingState != LivingState_UnderConstruction*/) {
            auto energyToTransfer = (atomicAdd(&otherCell->energy, 0) - cellMinEnergy) * cudaSimulationParameters.cellFunctionAttackerStrength[cell->color];
            if (energyToTransfer < 0) {
                return;
            }
            if (otherCell->genomeSize > cell->genomeSize) {
                auto genomeSizeBonus = cudaSimulationParameters.cellFunctionAttackerGenomeSizeBonus[cell->color][otherCell->color];
                energyToTransfer /= (1.0f + genomeSizeBonus * static_cast<float>(otherCell->genomeSize - cell->genomeSize));
            }
            if (otherCell->mutationId == cell->mutationId) {
                auto sameMutantPenalty =
                    cudaSimulationParameters.cellFunctionAttackerSameMutantPenalty[cell->color][otherCell->color];
                energyToTransfer *= (1.0f - sameMutantPenalty);
            }

            auto velocityPenalty = Math::length(cell->vel) * 20 * cudaSimulationParameters.cellFunctionAttackerVelocityPenalty[cell->color] + 1.0f;
            energyToTransfer /= velocityPenalty;

            auto numDefenderCells = countAndTrackDefenderCells(statistics, otherCell);
            float defendStrength =
                numDefenderCells == 0 ? 1.0f : powf(cudaSimulationParameters.cellFunctionDefenderAgainstAttackerStrength[cell->color], numDefenderCells);
            energyToTransfer /= defendStrength;

            if (!isHomogene(otherCell)) {
                energyToTransfer *= cudaSimulationParameters.cellFunctionAttackerColorInhomogeneityFactor[cell->color];
            }
            auto cellFunctionAttackerGeometryDeviationExponent = SpotCalculator::calcParameter(
                &SimulationParametersSpotValues::cellFunctionAttackerGeometryDeviationExponent,
                &SimulationParametersSpotActivatedValues::cellFunctionAttackerGeometryDeviationExponent,
                data,
                cell->absPos,
                cell->color);

            if (abs(cellFunctionAttackerGeometryDeviationExponent) > 0) {
                auto d = otherCell->absPos - cell->absPos;
                auto angle1 = calcOpenAngle(cell, d);
                auto angle2 = calcOpenAngle(otherCell, d * (-1));
                auto deviation = 1.0f - abs(360.0f - (angle1 + angle2)) / 360.0f;  //1 = no deviation, 0 = max deviation
                energyToTransfer *= powf(max(0.0f, min(1.0f, deviation)), cellFunctionAttackerGeometryDeviationExponent);
            }

            auto cellFunctionAttackerConnectionsMismatchPenalty = SpotCalculator::calcParameter(
                &SimulationParametersSpotValues::cellFunctionAttackerConnectionsMismatchPenalty,
                &SimulationParametersSpotActivatedValues::cellFunctionAttackerConnectionsMismatchPenalty,
                data,
                cell->absPos,
                cell->color);
            if (otherCell->numConnections > cell->numConnections + 1) {
                energyToTransfer *= (1.0f - cellFunctionAttackerConnectionsMismatchPenalty) * (1.0f - cellFunctionAttackerConnectionsMismatchPenalty);
            }
            if (otherCell->numConnections == cell->numConnections + 1) {
                energyToTransfer *= (1.0f - cellFunctionAttackerConnectionsMismatchPenalty);
            }

            auto color = calcMod(cell->color, MAX_COLORS);
            auto otherColor = calcMod(otherCell->color, MAX_COLORS);
            energyToTransfer *= SpotCalculator::calcParameter(
                &SimulationParametersSpotValues::cellFunctionAttackerFoodChainColorMatrix,
                &SimulationParametersSpotActivatedValues::cellFunctionAttackerFoodChainColorMatrix,
                data,
                cell->absPos,
                color,
                otherColor);

            if (abs(energyToTransfer) < NEAR_ZERO) {
                return;
            }

            //notify attacked cell
            if (otherCell->cellFunction != CellFunction_None) {
                atomicAdd(&otherCell->activity.channels[7], 1.0f);
            }

            someOtherCell = otherCell;
            if (energyToTransfer >= 0) {
                auto origEnergy = atomicAdd(&otherCell->energy, -energyToTransfer);
                if (origEnergy > cellMinEnergy + energyToTransfer) {
                    energyDelta += energyToTransfer;
                } else {
                    atomicAdd(&otherCell->energy, energyToTransfer);  //revert
                }
            } else {
                auto origEnergy = atomicAdd(&otherCell->energy, -energyToTransfer);
                if (origEnergy >= cellMinEnergy - (energyDelta + energyToTransfer)) {
                    energyDelta += energyToTransfer;
                } else {
                    atomicAdd(&otherCell->energy, energyToTransfer);  //revert
                }
            }
        }
    });

    if (energyDelta > NEAR_ZERO) {
        distributeEnergy(data, cell, energyDelta);
    } else {
        auto origEnergy = atomicAdd(&cell->energy, energyDelta);
        if (origEnergy + energyDelta < 0) {
            atomicAdd(&someOtherCell->energy, -energyDelta);  //revert
        }
    }

    radiate(data, cell);

    //output
    activity.channels[0] = energyDelta / 10;

    if (energyDelta > NEAR_ZERO) {
        statistics.incNumAttacks(cell->color);
    }

    CellFunctionProcessor::setActivity(cell, activity);
}

__device__ __inline__ void AttackerProcessor::radiate(SimulationData& data, Cell* cell)
{
    auto cellFunctionWeaponEnergyCost = SpotCalculator::calcParameter(
        &SimulationParametersSpotValues::cellFunctionAttackerEnergyCost,
        &SimulationParametersSpotActivatedValues::cellFunctionAttackerEnergyCost,
        data,
        cell->absPos,
        cell->color);
    if (cellFunctionWeaponEnergyCost > 0) {
        auto const cellEnergy = atomicAdd(&cell->energy, 0);

        auto const radiationEnergy = min(cellEnergy, cellFunctionWeaponEnergyCost);
        auto origEnergy = atomicAdd(&cell->energy, -radiationEnergy);
        if (origEnergy < 1.0f) {
            atomicAdd(&cell->energy, radiationEnergy);  //revert
            return;
        }

        float2 particleVel = (cell->vel * cudaSimulationParameters.radiationVelocityMultiplier)
            + float2{
                (data.numberGen1.random() - 0.5f) * cudaSimulationParameters.radiationVelocityPerturbation,
                (data.numberGen1.random() - 0.5f) * cudaSimulationParameters.radiationVelocityPerturbation};
        float2 particlePos = cell->absPos + Math::normalized(particleVel) * 1.5f - particleVel;
        data.cellMap.correctPosition(particlePos);

        ParticleProcessor::radiate(data, particlePos, particleVel, cell->color, radiationEnergy);
    }
}

__device__ __inline__ void AttackerProcessor::distributeEnergy(SimulationData& data, Cell* cell, float energyDelta)
{
    auto const& energyDistribution = cudaSimulationParameters.cellFunctionAttackerEnergyDistributionValue[cell->color];
    auto origEnergy = atomicAdd(&cell->energy, -energyDistribution);
    if (origEnergy > cudaSimulationParameters.cellNormalEnergy[cell->color]) {
        energyDelta += energyDistribution;
    } else {
        atomicAdd(&cell->energy, energyDistribution);  //revert
    }

    if (cell->cellFunctionData.attacker.mode == EnergyDistributionMode_ConnectedCells) {
        int numReceivers = cell->numConnections;
        for (int i = 0; i < cell->numConnections; ++i) {
            numReceivers += cell->connections[i].cell->numConnections;
        }
        float energyPerReceiver = energyDelta / (numReceivers + 1);

        for (int i = 0; i < cell->numConnections; ++i) {
            auto connectedCell = cell->connections[i].cell;
            atomicAdd(&connectedCell->energy, energyPerReceiver);
            energyDelta -= energyPerReceiver;
            for (int i = 0; i < connectedCell->numConnections; ++i) {
                auto connectedConnectedCell = connectedCell->connections[i].cell;
                atomicAdd(&connectedConnectedCell->energy, energyPerReceiver);
                energyDelta -= energyPerReceiver;
            }
        }
    }

    if (cell->cellFunctionData.attacker.mode == EnergyDistributionMode_TransmittersAndConstructors) {

        auto matchActiveConstructorFunc = [&](Cell* const& otherCell) {
            if (otherCell->livingState != LivingState_Ready) {
                return false;
            }
            if (otherCell->cellFunction == CellFunction_Constructor) {
                auto isFinished = GenomeDecoder::isFinishedSingleConstruction(otherCell->cellFunctionData.constructor);
                if (!isFinished && otherCell->creatureId == cell->creatureId) {
                    return true;
                }
            }
            return false;
        };
        auto matchTransmitterFunc = [&](Cell* const& otherCell) {
            if (otherCell->livingState != LivingState_Ready) {
                return false;
            }
            if (otherCell->cellFunction == CellFunction_Transmitter) {
                if (otherCell->creatureId == cell->creatureId) {
                    return true;
                }
            }
            return false;
        };

        Cell* receiverCells[20];
        int numReceivers;
        auto radius = cudaSimulationParameters.cellFunctionAttackerEnergyDistributionRadius[cell->color];
        data.cellMap.getMatchingCells(receiverCells, 20, numReceivers, cell->absPos, radius, cell->detached, matchActiveConstructorFunc);
        if (numReceivers == 0) {
            data.cellMap.getMatchingCells(receiverCells, 20, numReceivers, cell->absPos, radius, cell->detached, matchTransmitterFunc);
        }
        float energyPerReceiver = energyDelta / (numReceivers + 1);

        for (int i = 0; i < numReceivers; ++i) {
            auto receiverCell = receiverCells[i];
            atomicAdd(&receiverCell->energy, energyPerReceiver);
            energyDelta -= energyPerReceiver;
        }
    }
    atomicAdd(&cell->energy, energyDelta);
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

__inline__ __device__ int AttackerProcessor::countAndTrackDefenderCells(SimulationStatistics& statistics, Cell* cell)
{
    int result = 0;
    if (cell->cellFunction == CellFunction_None) {
        return result;
    }
    if (cell->cellFunction == CellFunction_Defender && cell->cellFunctionData.defender.mode == DefenderMode_DefendAgainstAttacker) {
        ++result;
    }
    for (int i = 0; i < cell->numConnections; ++i) {
        auto connectedCell = cell->connections[i].cell;
        if (connectedCell->cellFunction == CellFunction_Defender && connectedCell->cellFunctionData.defender.mode == DefenderMode_DefendAgainstAttacker) {
            statistics.incNumDefenderActivities(connectedCell->color);
            ++result;
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