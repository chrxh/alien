#pragma once


#include <EngineInterface/CellTypeConstants.h>

#include "ObjectConnectionProcessor.cuh"
#include "ConstantMemory.cuh"
#include "Entities.cuh"
#include "ParameterCalculator.cuh"
#include "EnergyProcessor.cuh"
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
    auto const& cell = &object->typeData.cell;
    if (SignalProcessor::isManuallyTriggered(data, object) && cell->rawEnergy < SimulationParameters::attackerMaxRawEnergyThreshold) {

        auto attackerEnergyCost = ParameterCalculator::calcParameter(cudaSimulationParameters.attackerEnergyCost, data, object->pos, object->color);
        auto cellMinEnergy = ParameterCalculator::calcParameter(cudaSimulationParameters.minCellEnergy, data, object->pos, object->color);
        if (cell->usableEnergy - attackerEnergyCost < cellMinEnergy) {
            cell->signal.channels[Channels::AttackerSuccess] = 0;
            return;
        }

        auto const& attackerMode = cell->cellTypeData.attacker.mode;

        auto sumEnergyToTransfer = 0.0f;
        data.objectMap.executeForEach(object->pos, cudaSimulationParameters.attackerRadius.value[object->color], object->detached, [&](auto const& otherObject) {
            if (otherObject->type == ObjectType_Structure) {
                return;
            }
            if (otherObject->fixed) {
                return;
            }

            // Only attack other cells which are in a visible cone with respect to the attack cell
            if (ObjectConnectionProcessor::existsOwnIntersectingObjectInBetween(data, object, otherObject)) {
                return;
            }

            // Attack Free Cell
            if (attackerMode == AttackerMode_FreeCell) {
                if (otherObject->type != ObjectType_FreeCell) {
                    return;
                }
                auto const& otherFreeCell = &otherObject->typeData.freeCell;

                // Filter by color restriction
                auto const& restrictToColor = cell->cellTypeData.attacker.modeData.attackFreeCell.restrictToColor;
                if (restrictToColor != 255 && otherObject->color != restrictToColor) {
                    return;
                }

                // Only attack cells with energy above base value
                auto energyToTransfer = atomicAdd(&otherFreeCell->rawEnergy, 0) * cudaSimulationParameters.attackerStrength.value[object->color];

                if (energyToTransfer < 0) {
                    return;
                }

                // Evaluate food chain color matrix
                auto color = calcMod(object->color, MAX_COLORS);
                auto otherColor = calcMod(otherObject->color, MAX_COLORS);
                energyToTransfer *=
                    ParameterCalculator::calcParameter(cudaSimulationParameters.attackerFoodChainColorMatrix, data, object->pos, color, otherColor);

                if (energyToTransfer > NEAR_ZERO) {
                    otherFreeCell->event = CellEvent_Attacked;
                    otherFreeCell->eventCounter = 10;
                    otherFreeCell->eventPos = object->pos;

                    // Absorb energy from attacked cell
                    auto origEnergy = atomicAdd(&otherFreeCell->rawEnergy, -energyToTransfer);
                    if (origEnergy > energyToTransfer) {
                        sumEnergyToTransfer += energyToTransfer;
                    }

                    // Revert
                    else {
                        atomicAdd(&otherFreeCell->rawEnergy, energyToTransfer);
                    }
                }
            }

            // Attack Cell
            else if (attackerMode == AttackerMode_Creature) {
                if (otherObject->type != ObjectType_Cell) {
                    return;
                }
                auto const& otherCell = &otherObject->typeData.cell;
                if (cell->isSameCreature(otherCell)) {
                    return;
                }
                // Do not attack offspring
                if (otherCell->creature->ancestorId == cell->creature->id) {
                    return;
                }

                // Filter by color restriction
                auto const& restrictToColor = cell->cellTypeData.attacker.modeData.attackCreature.restrictToColor;
                if (restrictToColor != 255 && otherObject->color != restrictToColor) {
                    return;
                }

                auto const& minNumCells = cell->cellTypeData.attacker.modeData.attackCreature.minNumCells;
                auto const& maxNumCells = cell->cellTypeData.attacker.modeData.attackCreature.maxNumCells;
                auto const& restrictToLineage = cell->cellTypeData.attacker.modeData.attackCreature.restrictToLineage;

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
                    if (restrictToLineage == LineageRestriction_SameLineage) {
                        if (cell->creature->lineageId != otherCell->creature->lineageId) {
                            return;
                        }
                    } else if (restrictToLineage == LineageRestriction_OtherLineage) {
                        if (cell->creature->lineageId == otherCell->creature->lineageId) {
                            return;
                        }
                    }
                }

                // Only attack cells with energy above base value
                auto energyToTransfer = atomicAdd(&otherCell->usableEnergy, 0) * cudaSimulationParameters.attackerStrength.value[object->color];

                if (energyToTransfer < 0) {
                    return;
                }

                auto color = calcMod(object->color, MAX_COLORS);
                auto otherColor = calcMod(otherObject->color, MAX_COLORS);

                // Evaluate defender strength
                auto numDefenderCells = countDefenderCells(statistics, otherObject);
                auto defendStrength =
                    numDefenderCells == 0 ? 1.0f : powf(cudaSimulationParameters.defenderAntiAttackerStrength.value[color] + 1.0f, numDefenderCells);
                energyToTransfer /= defendStrength;

                // Evaluate food chain color matrix
                energyToTransfer *=
                    ParameterCalculator::calcParameter(cudaSimulationParameters.attackerFoodChainColorMatrix, data, object->pos, color, otherColor);

                if (energyToTransfer > NEAR_ZERO) {

                    // Notify attacked cell

                    if (otherCell->signalState != SignalState_Active) {
                        SignalProcessor::createEmptySignal(otherObject);
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
            }
        });

        // Energy gain
        if (sumEnergyToTransfer > NEAR_ZERO) {
            atomicAdd(&cell->rawEnergy, sumEnergyToTransfer);

            cell->event = CellEvent_Attacking;
            cell->eventCounter = 6;
            statistics.incNumAttacks(object->color);
        }

        // Radiation
        if (attackerEnergyCost > 0) {
            EnergyProcessor::radiate(data, object, attackerEnergyCost);
        }

        // Output (signal is already present since attacker can only be manually triggered)
        cell->signal.channels[Channels::AttackerSuccess] = min(1.0f, max(0.0f, sumEnergyToTransfer / 10));
    }
}

__inline__ __device__ int AttackerProcessor::countDefenderCells(SimulationStatistics& statistics, Object* object)
{
    int result = 0;
    if (object->typeData.cell.cellType == CellType_Defender && object->typeData.cell.cellTypeData.defender.mode == DefenderMode_DefendAgainstAttacker) {
        ++result;
    }
    for (int i = 0; i < object->numConnections; ++i) {
        auto connectedObject = object->connections[i].object;
        if (connectedObject->type != ObjectType_Cell) {
            continue;
        }   
        if (connectedObject->typeData.cell.cellType == CellType_Defender && connectedObject->typeData.cell.cellTypeData.defender.mode == DefenderMode_DefendAgainstAttacker) {
            statistics.incNumDefenderActivities(connectedObject->color);
            ++result;
        }
    }
    return result;
}
