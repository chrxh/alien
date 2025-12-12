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
    if (SignalProcessor::isManuallyTriggered(data, cell) && cell->rawEnergy < SimulationParameters::attackerMaxRawEnergyThreshold) {
        float energyDelta = 0;
        auto cellMinEnergy = ParameterCalculator::calcParameter(cudaSimulationParameters.minCellEnergy, data, cell->pos, cell->color);
        auto baseValue = cellMinEnergy * 0.1f;

        data.cellMap.executeForEach(cell->pos, cudaSimulationParameters.attackerRadius.value[cell->color], cell->detached, [&](auto const& otherCell) {
            if (otherCell->creature != nullptr && otherCell->creature->id == cell->creature->id) {
                return;
            }
            if (otherCell->fixed) {
                return;
            }

            // Only attack cells with energy above base value
            auto energyToTransfer = (atomicAdd(&otherCell->usableEnergy, 0) - baseValue) * cudaSimulationParameters.attackerStrength.value[cell->color];
            if (energyToTransfer < 0) {
                return;
            }

            auto color = calcMod(cell->color, MAX_COLORS);
            auto otherColor = calcMod(otherCell->color, MAX_COLORS);

            // Evaluate defender strength
            auto numDefenderCells = countAndTrackDefenderCells(statistics, otherCell);
            float defendStrength =
                numDefenderCells == 0 ? 1.0f : powf(cudaSimulationParameters.defenderAntiAttackerStrength.value[color] + 1.0f, numDefenderCells);
            energyToTransfer /= defendStrength;

            // Evaluate color inhomogeneity factor
            if (!isHomogene(otherCell)) {
                energyToTransfer *= cudaSimulationParameters.attackerColorInhomogeneityFactor;
            }

            // Evaluate food chain color matrix
            energyToTransfer *= ParameterCalculator::calcParameter(cudaSimulationParameters.attackerFoodChainColorMatrix, data, cell->pos, color, otherColor);

            if (abs(energyToTransfer) < NEAR_ZERO) {
                return;
            }

            if (energyToTransfer > NEAR_ZERO) {

                // Notify attacked cell
                atomicAdd(&otherCell->signal.channels[Channels::AttackerNotify], 1.0f);
                otherCell->event = CellEvent_Attacked;
                otherCell->eventCounter = 10;
                otherCell->eventPos = cell->pos;

                auto origEnergy = atomicAdd(&otherCell->usableEnergy, -energyToTransfer);
                if (origEnergy > baseValue + energyToTransfer) {
                    energyDelta += energyToTransfer;
                }
                // Revert
                else {
                    atomicAdd(&otherCell->usableEnergy, energyToTransfer);
                }
            }
        });

        if (energyDelta > NEAR_ZERO) {
            atomicAdd(&cell->rawEnergy, energyDelta);
        }

        // Radiation
        auto cellTypeWeaponEnergyCost = ParameterCalculator::calcParameter(cudaSimulationParameters.attackerEnergyCost, data, cell->pos, cell->color);
        if (cellTypeWeaponEnergyCost > 0) {
            EnergyParticleProcessor::radiate(data, cell, cellTypeWeaponEnergyCost);
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
            if (color != otherOtherCell->color) {
                return false;
            }
        }
    }
    return true;
}