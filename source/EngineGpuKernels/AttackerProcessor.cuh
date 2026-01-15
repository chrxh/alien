#pragma once


#include <EngineInterface/CellTypeConstants.h>

#include "ObjectConnectionProcessor.cuh"
#include "ConstantMemory.cuh"
#include "Entity.cuh"
#include "ParameterCalculator.cuh"
#include "EnergyParticleProcessor.cuh"
#include "SignalProcessor.cuh"
#include "SimulationData.cuh"
#include "SimulationStatistics.cuh"

class AttackerProcessor
{
public:
    __inline__ __device__ static void process(SimulationData& data, SimulationStatistics& result);

private:
    __inline__ __device__ static void processCell(SimulationData& data, SimulationStatistics& statistics, Object* object);

    __inline__ __device__ static int countDefenderCells(SimulationStatistics& statistics, Object* object);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__device__ __inline__ void AttackerProcessor::process(SimulationData& data, SimulationStatistics& result)
{
    auto& operations = data.cellTypeOperations[CellType_Attacker];
    auto partition = calcSystemThreadPartition(operations.getNumEntries());
    for (int i = partition.startIndex; i <= partition.endIndex; i += partition.step) {
        processCell(data, result, operations.at(i).object);
    }
}

__device__ __inline__ void AttackerProcessor::processCell(SimulationData& data, SimulationStatistics& statistics, Object* object)
{
    if (SignalProcessor::isManuallyTriggered(data, object) && object->rawEnergy < SimulationParameters::attackerMaxRawEnergyThreshold) {

        auto attackerEnergyCost = ParameterCalculator::calcParameter(cudaSimulationParameters.attackerEnergyCost, data, object->pos, object->color);
        auto cellMinEnergy = ParameterCalculator::calcParameter(cudaSimulationParameters.minCellEnergy, data, object->pos, object->color);
        if (object->usableEnergy - attackerEnergyCost < cellMinEnergy) {
            object->signal.channels[Channels::AttackerSuccess] = 0;
            return;
        }

        auto const& attackerMode = object->cellTypeData.attacker.mode;

        auto sumEnergyToTransfer = 0.0f;
        data.cellMap.executeForEach(object->pos, cudaSimulationParameters.attackerRadius.value[object->color], object->detached, [&](auto const& otherCell) {

            if (attackerMode == AttackerMode_FreeCell) {
                if (otherCell->cellType != CellType_Free) {
                    return;
                }

                // Filter by color restriction
                auto const& restrictToColor = object->cellTypeData.attacker.modeData.attackFreeCell.restrictToColor;
                if (restrictToColor != 255 && otherCell->color != restrictToColor) {
                    return;
                }

            } else if (attackerMode == AttackerMode_Creature) {
                if (otherCell->creature == nullptr) {
                    return;
                }


                if (object->isSameCreature(otherCell)) {
                    return;
                }
                // Do not attack direct offspring (only applicable for Creature mode since free cells have no ancestry)
                if (object->creature != nullptr) {
                    if (otherCell->creature->ancestorId == object->creature->id) {
                        return;
                    }
                }
                if (otherCell->fixed) {
                    return;
                }

                // Filter by color restriction
                auto const& restrictToColor = object->cellTypeData.attacker.modeData.attackCreature.restrictToColor;
                if (restrictToColor != 255 && otherCell->color != restrictToColor) {
                    return;
                }

                auto const& minNumCells = object->cellTypeData.attacker.modeData.attackCreature.minNumCells;
                auto const& maxNumCells = object->cellTypeData.attacker.modeData.attackCreature.maxNumCells;
                auto const& restrictToLineage = object->cellTypeData.attacker.modeData.attackCreature.restrictToLineage;

                // Filter by minimum number of cells in creature
                if (minNumCells > 0 && otherCell->creature->numObjects < minNumCells) {
                    return;
                }

                // Filter by maximum number of cells in creature
                if (maxNumCells > 0 && otherCell->creature->numObjects > maxNumCells) {
                    return;
                }

                // Filter by lineage restriction
                if (restrictToLineage != LineageRestriction_No) {
                    if (object->creature == nullptr) {
                        return;
                    }
                    if (restrictToLineage == LineageRestriction_SameLineage) {
                        if (object->creature->lineageId != otherCell->creature->lineageId) {
                            return;
                        }
                    } else if (restrictToLineage == LineageRestriction_OtherLineage) {
                        if (object->creature->lineageId == otherCell->creature->lineageId) {
                            return;
                        }
                    }
                }
            }

            // Only attack cells with energy above base value
            auto energyToTransfer = atomicAdd(&otherCell->usableEnergy, 0) * cudaSimulationParameters.attackerStrength.value[object->color];
            if (energyToTransfer < 0) {
                return;
            }

            auto color = calcMod(object->color, MAX_COLORS);
            auto otherColor = calcMod(otherCell->color, MAX_COLORS);

            // Evaluate defender strength
            auto numDefenderCells = countDefenderCells(statistics, otherCell);
            auto defendStrength =
                numDefenderCells == 0 ? 1.0f : powf(cudaSimulationParameters.defenderAntiAttackerStrength.value[color] + 1.0f, numDefenderCells);
            energyToTransfer /= defendStrength;

            // Evaluate food chain color matrix
            energyToTransfer *= ParameterCalculator::calcParameter(cudaSimulationParameters.attackerFoodChainColorMatrix, data, object->pos, color, otherColor);

            if (energyToTransfer > NEAR_ZERO) {

                // Only attack other cells which are in a visible cone with respect to the attack cell
                if (ObjectConnectionProcessor::existsOwnIntersectingCellInBetween(data, object, otherCell)) {
                    return;
                }

                // Notify attacked cell
                if (otherCell->signalState != SignalState_Active) {
                    SignalProcessor::createEmptySignal(otherCell);
                }
                atomicAdd(&otherCell->signal.channels[Channels::AttackerNotify], 1.0f);
                otherCell->event = CellEvent_Attacked;
                otherCell->eventCounter = 10;
                otherCell->eventPos = object->pos;

                // Absorb energy from attacked cell
                auto origEnergy = atomicAdd(&otherCell->usableEnergy, -energyToTransfer);
                if (origEnergy > energyToTransfer) {
                    sumEnergyToTransfer += energyToTransfer;
                }
                // Revert
                else {
                    atomicAdd(&otherCell->usableEnergy, energyToTransfer);
                }
            }
        });

        // Energy gain
        if (sumEnergyToTransfer > NEAR_ZERO) {
            atomicAdd(&object->rawEnergy, sumEnergyToTransfer);

            object->event = CellEvent_Attacking;
            object->eventCounter = 6;
            statistics.incNumAttacks(object->color);
        }

        // Radiation
        if (attackerEnergyCost > 0) {
            EnergyParticleProcessor::radiate(data, object, attackerEnergyCost);
        }

        // Output (signal is already present since attacker can only be manually triggered)
        object->signal.channels[Channels::AttackerSuccess] = min(1.0f, max(0.0f, sumEnergyToTransfer / 10));
    }
}

__inline__ __device__ int AttackerProcessor::countDefenderCells(SimulationStatistics& statistics, Object* object)
{
    int result = 0;
    if (object->cellType == CellType_Defender && object->cellTypeData.defender.mode == DefenderMode_DefendAgainstAttacker) {
        ++result;
    }
    for (int i = 0; i < object->numConnections; ++i) {
        auto connectedCell = object->connections[i].object;
        if (connectedCell->cellType == CellType_Defender && connectedCell->cellTypeData.defender.mode == DefenderMode_DefendAgainstAttacker) {
            statistics.incNumDefenderActivities(connectedCell->color);
            ++result;
        }
    }
    return result;
}
