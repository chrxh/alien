#pragma once


#include <EngineInterface/CellTypeConstants.h>

#include "ObjectConnectionProcessor.cuh"
#include "ConstantMemory.cuh"
#include "Entity.cuh"
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
    if (SignalProcessor::isManuallyTriggered(data, object) && object->typeData.cell.rawEnergy < SimulationParameters::attackerMaxRawEnergyThreshold) {

        auto attackerEnergyCost = ParameterCalculator::calcParameter(cudaSimulationParameters.attackerEnergyCost, data, object->pos, object->color);
        auto cellMinEnergy = ParameterCalculator::calcParameter(cudaSimulationParameters.minCellEnergy, data, object->pos, object->color);
        if (object->typeData.cell.usableEnergy - attackerEnergyCost < cellMinEnergy) {
            object->typeData.cell.signal.channels[Channels::AttackerSuccess] = 0;
            return;
        }

        auto const& attackerMode = object->typeData.cell.cellTypeData.attacker.mode;

        auto sumEnergyToTransfer = 0.0f;
        data.objectMap.executeForEach(object->pos, cudaSimulationParameters.attackerRadius.value[object->color], object->detached, [&](auto const& otherObject) {

            if (attackerMode == AttackerMode_FreeCell) {
                if (otherObject->typeData.cell.cellType != CellType_Free) {
                    return;
                }

                // Filter by color restriction
                auto const& restrictToColor = object->typeData.cell.cellTypeData.attacker.modeData.attackFreeCell.restrictToColor;
                if (restrictToColor != 255 && otherObject->color != restrictToColor) {
                    return;
                }

            } else if (attackerMode == AttackerMode_Creature) {
                if (otherObject->typeData.cell.creature == nullptr) {
                    return;
                }


                if (object->typeData.cell.isSameCreature(&otherObject->typeData.cell)) {
                    return;
                }
                // Do not attack direct offspring (only applicable for Creature mode since free cells have no ancestry)
                if (object->typeData.cell.creature != nullptr) {
                    if (otherObject->typeData.cell.creature->ancestorId == object->typeData.cell.creature->id) {
                        return;
                    }
                }
                if (otherObject->fixed) {
                    return;
                }

                // Filter by color restriction
                auto const& restrictToColor = object->typeData.cell.cellTypeData.attacker.modeData.attackCreature.restrictToColor;
                if (restrictToColor != 255 && otherObject->color != restrictToColor) {
                    return;
                }

                auto const& minNumCells = object->typeData.cell.cellTypeData.attacker.modeData.attackCreature.minNumCells;
                auto const& maxNumCells = object->typeData.cell.cellTypeData.attacker.modeData.attackCreature.maxNumCells;
                auto const& restrictToLineage = object->typeData.cell.cellTypeData.attacker.modeData.attackCreature.restrictToLineage;

                // Filter by minimum number of cells in creature
                if (minNumCells > 0 && otherObject->typeData.cell.creature->numObjects < minNumCells) {
                    return;
                }

                // Filter by maximum number of cells in creature
                if (maxNumCells > 0 && otherObject->typeData.cell.creature->numObjects > maxNumCells) {
                    return;
                }

                // Filter by lineage restriction
                if (restrictToLineage != LineageRestriction_No) {
                    if (object->typeData.cell.creature == nullptr) {
                        return;
                    }
                    if (restrictToLineage == LineageRestriction_SameLineage) {
                        if (object->typeData.cell.creature->lineageId != otherObject->typeData.cell.creature->lineageId) {
                            return;
                        }
                    } else if (restrictToLineage == LineageRestriction_OtherLineage) {
                        if (object->typeData.cell.creature->lineageId == otherObject->typeData.cell.creature->lineageId) {
                            return;
                        }
                    }
                }
            }

            // Only attack cells with energy above base value
            auto energyToTransfer = atomicAdd(&otherObject->typeData.cell.usableEnergy, 0) * cudaSimulationParameters.attackerStrength.value[object->color];
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
            energyToTransfer *= ParameterCalculator::calcParameter(cudaSimulationParameters.attackerFoodChainColorMatrix, data, object->pos, color, otherColor);

            if (energyToTransfer > NEAR_ZERO) {

                // Only attack other cells which are in a visible cone with respect to the attack cell
                if (ObjectConnectionProcessor::existsOwnIntersectingCellInBetween(data, object, otherObject)) {
                    return;
                }

                // Notify attacked cell
                if (otherObject->typeData.cell.signalState != SignalState_Active) {
                    SignalProcessor::createEmptySignal(otherObject);
                }
                atomicAdd(&otherObject->typeData.cell.signal.channels[Channels::AttackerNotify], 1.0f);
                otherObject->typeData.cell.event = CellEvent_Attacked;
                otherObject->typeData.cell.eventCounter = 10;
                otherObject->typeData.cell.eventPos = object->pos;

                // Absorb energy from attacked cell
                auto origEnergy = atomicAdd(&otherObject->typeData.cell.usableEnergy, -energyToTransfer);
                if (origEnergy > energyToTransfer) {
                    sumEnergyToTransfer += energyToTransfer;
                }
                // Revert
                else {
                    atomicAdd(&otherObject->typeData.cell.usableEnergy, energyToTransfer);
                }
            }
        });

        // Energy gain
        if (sumEnergyToTransfer > NEAR_ZERO) {
            atomicAdd(&object->typeData.cell.rawEnergy, sumEnergyToTransfer);

            object->typeData.cell.event = CellEvent_Attacking;
            object->typeData.cell.eventCounter = 6;
            statistics.incNumAttacks(object->color);
        }

        // Radiation
        if (attackerEnergyCost > 0) {
            EnergyProcessor::radiate(data, object, attackerEnergyCost);
        }

        // Output (signal is already present since attacker can only be manually triggered)
        object->typeData.cell.signal.channels[Channels::AttackerSuccess] = min(1.0f, max(0.0f, sumEnergyToTransfer / 10));
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
        if (connectedObject->typeData.cell.cellType == CellType_Defender && connectedObject->typeData.cell.cellTypeData.defender.mode == DefenderMode_DefendAgainstAttacker) {
            statistics.incNumDefenderActivities(connectedObject->color);
            ++result;
        }
    }
    return result;
}
