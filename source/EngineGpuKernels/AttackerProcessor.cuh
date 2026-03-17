#pragma once


#include <EngineInterface/CellTypeConstants.h>
#include <EngineInterface/SimulationParameters.h>

#include "ConstantMemory.cuh"
#include "EnergyProcessor.cuh"
#include "Entities.cuh"
#include "NeuronProcessor.cuh"
#include "ObjectConnectionProcessor.cuh"
#include "ParameterCalculator.cuh"
#include "SimulationData.cuh"
#include "SimulationStatistics.cuh"

class AttackerProcessor
{
public:
    __inline__ __device__ static void process(SimulationData& data, SimulationStatistics& result);

private:
    __inline__ __device__ static void processCell(SimulationData& data, SimulationStatistics& statistics, Object* object);

    __inline__ __device__ static int countDefenderCells(SimulationStatistics& statistics, Object* object);

    __inline__ __device__ static bool
    isContainedInSensorMatches(uint64_t const* sensorTargetCreatureIds, uint16_t const* sensorRestrictToColors, int numSensorTargets, uint64_t creatureId, int color);

    static constexpr int MaxSensorTargets = 8;
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
    if (object->fixed) {
        return;
    }
    auto const& cell = &object->typeData.cell;
    if (NeuronProcessor::isManuallyTriggered(data, object) && cell->rawEnergy < SimulationParameters::attackerMaxRawEnergyThreshold) {

        auto attackerEnergyCost = ParameterCalculator::calcParameter(cudaSimulationParameters.attackerEnergyCost, data, object->pos, object->color);
        auto cellMinEnergy = ParameterCalculator::calcParameter(cudaSimulationParameters.minCellEnergy, data, object->pos, object->color);
        if (cell->usableEnergy - attackerEnergyCost < cellMinEnergy) {
            cell->signal.channels[Channels::AttackerSuccess] = 0;
            return;
        }

        auto const& attackerMode = cell->cellTypeData.attacker.mode;

        // For AttackCreature mode: collect creatureIds and color restrictions from sensor lastMatches in vicinity
        uint64_t sensorTargetCreatureIds[MaxSensorTargets];
        uint16_t sensorRestrictToColors[MaxSensorTargets];
        int numSensorTargets = 0;
        if (attackerMode == AttackerMode_Creature) {
            auto creatureId = cell->creature->id;
            data.objectMap.executeForEach(object->pos, SimulationParameters::attackerCreatureSensorRange, object->detached, [&](auto const& nearObject) {
                if (nearObject->type != ObjectType_Cell) {
                    return;
                }
                auto const& nearCell = &nearObject->typeData.cell;
                // Only consider sensor cells from the same creature
                if (nearCell->creature->id != creatureId) {
                    return;
                }
                if (nearCell->cellType != CellType_Sensor) {
                    return;
                }
                // Check if sensor has a valid lastMatch
                if (!nearCell->cellTypeData.sensor.lastMatchAvailable) {
                    return;
                }
                // Only use lastMatch if sensor is tagged for attackers
                if (!nearCell->cellTypeData.sensor.tagForAttackers) {
                    return;
                }
                auto matchCreatureId = nearCell->cellTypeData.sensor.lastMatch.creatureIdPart;

                // Get the color restriction from the sensor's DetectCreature mode
                uint16_t restrictToColors = 0x3ff;
                if (nearCell->cellTypeData.sensor.mode == SensorMode_DetectCreature) {
                    restrictToColors = nearCell->cellTypeData.sensor.modeData.detectCreature.restrictToColors;
                }

                // If creatureId already in list, merge color restrictions
                for (int i = 0; i < numSensorTargets; ++i) {
                    if (sensorTargetCreatureIds[i] == matchCreatureId) {
                        sensorRestrictToColors[i] |= restrictToColors;
                        return;
                    }
                }
                if (numSensorTargets < MaxSensorTargets) {
                    sensorTargetCreatureIds[numSensorTargets] = matchCreatureId;
                    sensorRestrictToColors[numSensorTargets] = restrictToColors;
                    ++numSensorTargets;
                }
            });
        }

        auto sumEnergyToTransfer = 0.0f;
        data.objectMap.executeForEach(
            object->pos, cudaSimulationParameters.attackerRadius.value[object->color], object->detached, [&](auto const& otherObject) {
                if (otherObject->type == ObjectType_Structure || otherObject->type == ObjectType_Fluid) {
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
                    auto const& restrictToColors = cell->cellTypeData.attacker.modeData.attackFreeCell.restrictToColors;
                    if (!((restrictToColors >> otherObject->color) & 1)) {
                        return;
                    }

                    // Calc base energy for attacking
                    auto energyToTransfer =
                        atomicAdd(&otherFreeCell->energy, 0) * cudaSimulationParameters.attackerStrength.value[object->color] * TIMESTEPS_PER_CELL_FUNCTION;

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
                        auto origEnergy = atomicAdd(&otherFreeCell->energy, -energyToTransfer);
                        if (origEnergy > energyToTransfer) {
                            sumEnergyToTransfer += energyToTransfer;
                        }

                        // Revert
                        else {
                            atomicAdd(&otherFreeCell->energy, energyToTransfer);
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

                    // Check if target creature is in the list of sensor-detected targets (including color restriction)
                    auto otherCreatureId = otherCell->creature->id;
                    if (!isContainedInSensorMatches(sensorTargetCreatureIds, sensorRestrictToColors, numSensorTargets, otherCreatureId, otherObject->color)) {
                        return;
                    }

                    // Calculate energy gain
                    auto energyToTransfer =
                        atomicAdd(&otherCell->usableEnergy, 0) * cudaSimulationParameters.attackerStrength.value[object->color] * TIMESTEPS_PER_CELL_FUNCTION;

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

                    // Evaluate lineage
                    if (cell->creature->genome->isRelatedLineage(otherCell->creature->genome)) {
                        energyToTransfer *= (1.0f - cudaSimulationParameters.attackerRelatedLineageProtection.value[object->color]);
                    }

                    if (energyToTransfer > NEAR_ZERO) {

                        // Notify attacked cell
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
        if (connectedObject->typeData.cell.cellType == CellType_Defender
            && connectedObject->typeData.cell.cellTypeData.defender.mode == DefenderMode_DefendAgainstAttacker) {
            statistics.incNumDefenderActivities(connectedObject->color);
            ++result;
        }
    }
    return result;
}

__inline__ __device__ bool
AttackerProcessor::isContainedInSensorMatches(uint64_t const* sensorTargetCreatureIds, uint16_t const* sensorRestrictToColors, int numSensorTargets, uint64_t creatureId, int color)
{
    // The sensor stores only the lower 16 bits of the creatureId (creatureIdPart)
    auto creatureIdPart = creatureId & 0xffff;
    for (int i = 0; i < numSensorTargets; ++i) {
        if (sensorTargetCreatureIds[i] == creatureIdPart && ((sensorRestrictToColors[i] >> color) & 1)) {
            return true;
        }
    }
    return false;
}
