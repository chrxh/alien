#pragma once


#include <EngineInterface/CellTypeConstants.h>

#include "ConstantMemory.cuh"
#include "Object.cuh"
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
    __inline__ __device__ static void processCell(SimulationData& data, SimulationStatistics& statistics, Cell* cell);

    __inline__ __device__ static bool existsOwnIntersectingCellInBetween(SimulationData& data, Cell* cell, Cell* otherCell);
    __inline__ __device__ static int countAndTrackDefenderCells(SimulationStatistics& statistics, Cell* cell);
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
    if (SignalProcessor::isManuallyTriggered(data, cell) && cell->rawEnergy < SimulationParameters::attackerMaxRawEnergyThreshold) {
        auto sumEnergyToTransfer = 0.0f;
        data.cellMap.executeForEach(cell->pos, cudaSimulationParameters.attackerRadius.value[cell->color], cell->detached, [&](auto const& otherCell) {
            if (otherCell->creature == nullptr) {
                return;
            }
            if (cell->isSameCreature(otherCell)) {
                return;
            }
            // Do not attack direct offspring
            if (otherCell->creature->ancestorId == cell->creature->id) {
                return;
            }
            if (otherCell->fixed) {
                return;
            }

            // Only attack cells with energy above base value
            auto energyToTransfer = atomicAdd(&otherCell->usableEnergy, 0) * cudaSimulationParameters.attackerStrength.value[cell->color];
            if (energyToTransfer < 0) {
                return;
            }

            auto color = calcMod(cell->color, MAX_COLORS);
            auto otherColor = calcMod(otherCell->color, MAX_COLORS);

            // Evaluate defender strength
            auto numDefenderCells = countAndTrackDefenderCells(statistics, otherCell);
            auto defendStrength =
                numDefenderCells == 0 ? 1.0f : powf(cudaSimulationParameters.defenderAntiAttackerStrength.value[color] + 1.0f, numDefenderCells);
            energyToTransfer /= defendStrength;

            // Evaluate food chain color matrix
            energyToTransfer *= ParameterCalculator::calcParameter(cudaSimulationParameters.attackerFoodChainColorMatrix, data, cell->pos, color, otherColor);

            if (energyToTransfer > NEAR_ZERO) {

                // Only attack other cells which are in a visible cone with respect to the attack cell
                if (existsOwnIntersectingCellInBetween(data, cell, otherCell)) {
                    return;
                }

                // Notify attacked cell
                if (!otherCell->signal.active) {
                    SignalProcessor::createEmptySignal(otherCell);
                }
                atomicAdd(&otherCell->signal.channels[Channels::AttackerNotify], 1.0f);
                otherCell->event = CellEvent_Attacked;
                otherCell->eventCounter = 10;
                otherCell->eventPos = cell->pos;

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
            atomicAdd(&cell->rawEnergy, sumEnergyToTransfer);

            cell->event = CellEvent_Attacking;
            cell->eventCounter = 6;
            statistics.incNumAttacks(cell->color);
        }

        // Radiation
        auto cellTypeWeaponEnergyCost = ParameterCalculator::calcParameter(cudaSimulationParameters.attackerEnergyCost, data, cell->pos, cell->color);
        if (cellTypeWeaponEnergyCost > 0) {
            EnergyParticleProcessor::radiate(data, cell, cellTypeWeaponEnergyCost);
        }

        // Output (signal is already present since attacker can only be manually triggered)
        cell->signal.channels[Channels::AttackerSuccess] = min(1.0f, max(0.0f, sumEnergyToTransfer / 10));
    }
}

__inline__ __device__ bool AttackerProcessor::existsOwnIntersectingCellInBetween(SimulationData& data, Cell* cell, Cell* otherCell)
{
    auto result = false;
    data.cellMap.executeForEach(cell->pos, cudaSimulationParameters.attackerRadius.value[cell->color], cell->detached, [&](Cell* nearCell) {
        if (result) {
            return;
        }
        if (nearCell == cell) {
            return;
        }
        if (nearCell == otherCell) {
            return;
        }
        if (!cell->isSameCreature(nearCell)) {
            return;
        }
        for (int i = 0; i < nearCell->numConnections; ++i) {
            auto connectedNearCell = nearCell->connections[i].cell;
            if (Math::crossing(nearCell->pos, connectedNearCell->pos, cell->pos, otherCell->pos)) {
                result = true;
                return;
            }
        }
    });
    return result;
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
