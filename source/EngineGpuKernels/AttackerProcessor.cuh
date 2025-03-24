#pragma once


#include "EngineInterface/CellTypeConstants.h"

#include "Object.cuh"
#include "SignalProcessor.cuh"
#include "ConstantMemory.cuh"
#include "SimulationData.cuh"
#include "ZoneCalculator.cuh"
#include "SimulationStatistics.cuh"
#include "RadiationProcessor.cuh"

class AttackerProcessor
{
public:
    __inline__ __device__ static void process(SimulationData& data, SimulationStatistics& result);

private:
    __inline__ __device__ static void processCell(SimulationData& data, SimulationStatistics& statistics, Cell* cell);

    __inline__ __device__ static float calcOpenAngle(Cell* cell, float2 direction);
    __inline__ __device__ static int countAndTrackDefenderCells(SimulationStatistics& statistics, Cell* cell);
    __inline__ __device__ static bool isHomogene(Cell* cell);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__device__ __inline__ void AttackerProcessor::process(SimulationData& data, SimulationStatistics& result)
{
    auto& operations = data.cellTypeOperations[CellType_Attacker];
    auto partition = calcAllThreadsPartition(operations.getNumEntries());
    for (int i = partition.startIndex; i <= partition.endIndex; ++i) {
        processCell(data, result, operations.at(i).cell);
    }
}

__device__ __inline__ void AttackerProcessor::processCell(SimulationData& data, SimulationStatistics& statistics, Cell* cell)
{
    if (SignalProcessor::isManuallyTriggered(data, cell)) {
        float energyDelta = 0;
        auto cellMinEnergy = ZoneCalculator::calcParameter(
            &SimulationParametersZoneValues::minCellEnergy, &SimulationParametersZoneActivatedValues::cellMinEnergy, data, cell->pos, cell->color);
        auto baseValue = cudaSimulationParameters.attackerDestroyCells ? cellMinEnergy * 0.1f : cellMinEnergy;

        Cell* someOtherCell = nullptr;
        data.cellMap.executeForEach(cell->pos, cudaSimulationParameters.attackerRadius[cell->color], cell->detached, [&](auto const& otherCell) {
            if (cell->creatureId != 0 && otherCell->creatureId == cell->creatureId) {
                return;
            }
            if (cell->creatureId == 0 && CellConnectionProcessor::isConnectedConnected(cell, otherCell)) {
                return;
            }
            if (otherCell->barrier) {
                return;
            }

            // Only attack cells with energy above base value
            auto energyToTransfer = (atomicAdd(&otherCell->energy, 0) - baseValue) * cudaSimulationParameters.attackerStrength[cell->color];
            if (energyToTransfer < 0) {
                return;
            }

            auto color = calcMod(cell->color, MAX_COLORS);
            auto otherColor = calcMod(otherCell->color, MAX_COLORS);

            // Evaluate sensor detection factor
            if (cudaSimulationParameters.features.advancedAttackerControl && otherCell->detectedByCreatureId != (cell->creatureId & 0xffff)) {
                energyToTransfer *= (1.0f - cudaSimulationParameters.attackerSensorDetectionFactor[color]);
            }

            // Evaluate genome complexity bonus
            if (otherCell->genomeComplexity > cell->genomeComplexity) {
                auto cellTypeAttackerGenomeComplexityBonus = ZoneCalculator::calcParameter(
                    &SimulationParametersZoneValues::cellTypeAttackerGenomeComplexityBonus,
                    &SimulationParametersZoneActivatedValues::cellTypeAttackerGenomeComplexityBonus,
                    data,
                    cell->pos,
                    color,
                    otherColor);
                energyToTransfer /=
                    (1.0f + cellTypeAttackerGenomeComplexityBonus * (otherCell->genomeComplexity - cell->genomeComplexity));
            }

            // Evaluate same mutant penalty
            if (cudaSimulationParameters.features.advancedAttackerControl
                && ((otherCell->mutationId == cell->mutationId) || (otherCell->ancestorMutationId == static_cast<uint8_t>(cell->mutationId & 0xff)))
                && cell->mutationId != 0) {
                energyToTransfer *= (1.0f - cudaSimulationParameters.attackerSameMutantPenalty[color][otherColor]);
            }

            // Evaluate new complex mutant penalty
            if (cudaSimulationParameters.features.advancedAttackerControl && cell->mutationId < otherCell->mutationId
                && cell->genomeComplexity <= otherCell->genomeComplexity) {
                auto cellTypeAttackerArisingComplexMutantPenalty = ZoneCalculator::calcParameter(
                    &SimulationParametersZoneValues::cellTypeAttackerNewComplexMutantPenalty,
                    &SimulationParametersZoneActivatedValues::cellTypeAttackerNewComplexMutantPenalty,
                    data,
                    cell->pos,
                    color,
                    otherColor);
                energyToTransfer *= (1.0f - cellTypeAttackerArisingComplexMutantPenalty);
            }

            // Evaluate defender strength
            auto numDefenderCells = countAndTrackDefenderCells(statistics, otherCell);
            float defendStrength =
                numDefenderCells == 0 ? 1.0f : powf(cudaSimulationParameters.defenderAntiAttackerStrength[color] + 1.0f, numDefenderCells);
            energyToTransfer /= defendStrength;

            // Evaluate color inhomogeneity factor
            if (!isHomogene(otherCell)) {
                energyToTransfer *= cudaSimulationParameters.attackerColorInhomogeneityFactor;
            }

            // Evaluate geometry deviation
            if (cudaSimulationParameters.features.advancedAttackerControl) {
                auto cellTypeAttackerGeometryDeviationExponent = ZoneCalculator::calcParameter(
                    &SimulationParametersZoneValues::cellTypeAttackerGeometryDeviationExponent,
                    &SimulationParametersZoneActivatedValues::cellTypeAttackerGeometryDeviationExponent,
                    data,
                    cell->pos,
                    cell->color);

                if (abs(cellTypeAttackerGeometryDeviationExponent) > 0) {
                    auto d = otherCell->pos - cell->pos;
                    auto angle1 = calcOpenAngle(cell, d);
                    auto angle2 = calcOpenAngle(otherCell, d * (-1));
                    auto deviation = 1.0f - abs(360.0f - (angle1 + angle2)) / 360.0f;  //1 = no deviation, 0 = max deviation
                    energyToTransfer *= powf(max(0.0f, min(1.0f, deviation)), cellTypeAttackerGeometryDeviationExponent);
                }
            }

            // Evaluate attacker connections mismatch penalty
            if (cudaSimulationParameters.features.advancedAttackerControl) {
                auto cellTypeAttackerConnectionsMismatchPenalty = ZoneCalculator::calcParameter(
                    &SimulationParametersZoneValues::cellTypeAttackerConnectionsMismatchPenalty,
                    &SimulationParametersZoneActivatedValues::cellTypeAttackerConnectionsMismatchPenalty,
                    data,
                    cell->pos,
                    cell->color);
                if (otherCell->numConnections > cell->numConnections + 1) {
                    energyToTransfer *= (1.0f - cellTypeAttackerConnectionsMismatchPenalty) * (1.0f - cellTypeAttackerConnectionsMismatchPenalty);
                }
                if (otherCell->numConnections == cell->numConnections + 1) {
                    energyToTransfer *= (1.0f - cellTypeAttackerConnectionsMismatchPenalty);
                }
            }

            // Evaluate food chain color matrix
            energyToTransfer *= ZoneCalculator::calcParameter(
                &SimulationParametersZoneValues::cellTypeAttackerFoodChainColorMatrix,
                &SimulationParametersZoneActivatedValues::cellTypeAttackerFoodChainColorMatrix,
                data,
                cell->pos,
                color,
                otherColor);

            if (abs(energyToTransfer) < NEAR_ZERO) {
                return;
            }

            someOtherCell = otherCell;
            if (energyToTransfer > NEAR_ZERO) {

                // Notify attacked cell
                atomicAdd(&otherCell->signal.channels[Channels::AttackerNotify], 1.0f);
                otherCell->event = CellEvent_Attacked;
                otherCell->eventCounter = 6;
                otherCell->eventPos = cell->pos;

                auto origEnergy = atomicAdd(&otherCell->energy, -energyToTransfer);
                if (origEnergy > baseValue + energyToTransfer) {
                    energyDelta += energyToTransfer;
                }
                // Revert
                else {
                    atomicAdd(&otherCell->energy, energyToTransfer);
                }
            } else if (energyToTransfer < -NEAR_ZERO) {
                auto origEnergy = atomicAdd(&otherCell->energy, -energyToTransfer);
                if (origEnergy >= baseValue - (energyDelta + energyToTransfer)) {
                    energyDelta += energyToTransfer;
                }
                // Revert
                else {
                    atomicAdd(&otherCell->energy, energyToTransfer);
                }
            }
        });

        if (energyDelta > NEAR_ZERO) {
            atomicAdd(&cell->energy, energyDelta);
            //distributeEnergy(data, cell, energyDelta);
        } else {
            auto origEnergy = atomicAdd(&cell->energy, energyDelta);
            if (origEnergy + energyDelta < 0) {
                atomicAdd(&someOtherCell->energy, -energyDelta);  //revert
            }
        }

        // Radiation
        auto cellTypeWeaponEnergyCost = ZoneCalculator::calcParameter(
            &SimulationParametersZoneValues::cellTypeAttackerEnergyCost,
            &SimulationParametersZoneActivatedValues::cellTypeAttackerEnergyCost,
            data,
            cell->pos,
            cell->color);
        if (cellTypeWeaponEnergyCost > 0) {
            RadiationProcessor::radiate(data, cell, cellTypeWeaponEnergyCost);
        }

        // Output
        cell->signal.channels[Channels::AttackerSuccess] = min(1.0f, max(0.0f, energyDelta / 10));

        if (energyDelta > NEAR_ZERO) {
            cell->event = CellEvent_Attacking;
            cell->eventCounter = 6;
            statistics.incNumAttacks(cell->color);
        }
    }
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

    float largerAngle = Math::angleOfVector(cell->connections[0].cell->pos - cell->pos);
    float smallerAngle = largerAngle;

    for (int i = 1; i < cell->numConnections; ++i) {
        auto otherCell = cell->connections[i].cell;
        auto angle = Math::angleOfVector(otherCell->pos - cell->pos);
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

__inline__ __device__ int AttackerProcessor::countAndTrackDefenderCells(SimulationStatistics& statistics, Cell* cell)
{
    int result = 0;
    if (cell->cellType == CellType_Defender && cell->cellTypeData.defender.mode == DefenderMode_DefendAgainstAttacker) {
        ++result;
    }
    for (int i = 0; i < cell->numConnections; ++i) {
        auto connectedCell = cell->connections[i].cell;
        if (connectedCell->cellType == CellType_Defender && connectedCell->cellTypeData.defender.mode == DefenderMode_DefendAgainstAttacker) {
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