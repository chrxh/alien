#pragma once

#include "EngineInterface/CellTypeConstants.h"

#include "Object.cuh"
#include "SignalProcessor.cuh"
#include "ConstantMemory.cuh"
#include "SimulationData.cuh"
#include "SimulationStatistics.cuh"

class TransmitterProcessor
{
public:
    __inline__ __device__ static void process(SimulationData& data, SimulationStatistics& statistics);

private:
    __inline__ __device__ static void processCell(SimulationData& data, SimulationStatistics& statistics, Cell* cell);
    __inline__ __device__ static void distributeEnergy(SimulationData& data, SimulationStatistics& statistics, Cell* cell);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__device__ __inline__ void TransmitterProcessor::process(SimulationData& data, SimulationStatistics& statistics)
{
    auto& operations = data.cellTypeOperations[CellType_Depot];
    auto partition = calcAllThreadsPartition(operations.getNumEntries());
    for (int i = partition.startIndex; i <= partition.endIndex; ++i) {
        auto const& cell = operations.at(i).cell;
        processCell(data, statistics, cell);
    }
}

__device__ __inline__ void TransmitterProcessor::processCell(SimulationData& data, SimulationStatistics& statistics, Cell* cell)
{
    distributeEnergy(data, statistics, cell);
}

__device__ __inline__ void TransmitterProcessor::distributeEnergy(SimulationData& data, SimulationStatistics& statistics, Cell* cell)
{
    float energyDelta = 0;
    auto const& energyDistribution = cudaSimulationParameters.transmitterEnergyDistributionValue.value[cell->color];
    auto origEnergy = atomicAdd(&cell->energy, -energyDistribution);
    if (origEnergy > cudaSimulationParameters.normalCellEnergy.value[cell->color]) {
        energyDelta = energyDistribution;
    } else {
        atomicAdd(&cell->energy, energyDistribution);  //revert
    }
    for (int i = 0; i < cell->numConnections; ++i) {
        auto connectedCell = cell->connections[i].cell;
        if (connectedCell->cellType == CellType_Depot) {
            continue;
        }
        if (connectedCell->cellType == CellType_Constructor) {
            if (!GenomeDecoder::isFinished(connectedCell->cellTypeData.constructor)) {
                continue;
            }
        }
        auto origEnergy = atomicAdd(&connectedCell->energy, -energyDistribution);
        if (origEnergy > cudaSimulationParameters.normalCellEnergy.value[cell->color]) {
            energyDelta += energyDistribution;
        } else {
            atomicAdd(&connectedCell->energy, energyDistribution);  //revert
        }
    }

    if (energyDelta > NEAR_ZERO) {
        statistics.incNumTransmitterActivities(cell->color);
    }

    if (cell->cellTypeData.transmitter.mode == EnergyDistributionMode_ConnectedCells) {
        int numReceivers = cell->numConnections;
        for (int i = 0; i < cell->numConnections; ++i) {
            numReceivers += cell->connections[i].cell->numConnections;
        }
        float energyPerReceiver = energyDelta / (numReceivers + 1);

        for (int i = 0; i < cell->numConnections; ++i) {
            auto connectedCell = cell->connections[i].cell;
            if (cudaSimulationParameters.transmitterEnergyDistributionSameCreature.value && connectedCell->creatureId != cell->creatureId) {
                continue;
            }
            atomicAdd(&connectedCell->energy, energyPerReceiver);

            energyDelta -= energyPerReceiver;
            for (int i = 0; i < connectedCell->numConnections; ++i) {
                auto connectedConnectedCell = connectedCell->connections[i].cell;
                if (cudaSimulationParameters.transmitterEnergyDistributionSameCreature.value
                    && connectedConnectedCell->creatureId != cell->creatureId) {
                    continue;
                }
                atomicAdd(&connectedConnectedCell->energy, energyPerReceiver);
                energyDelta -= energyPerReceiver;
            }
        }
    }
    if (cell->cellTypeData.transmitter.mode == EnergyDistributionMode_TransmittersAndConstructors) {
        auto matchActiveConstructorFunc = [&](Cell* const& otherCell) {
            if (otherCell->livingState != LivingState_Ready) {
                return false;
            }
            if (otherCell->cellType == CellType_Constructor) {
                if (!GenomeDecoder::isFinished(otherCell->cellTypeData.constructor)
                    && (!cudaSimulationParameters.transmitterEnergyDistributionSameCreature.value || otherCell->creatureId == cell->creatureId)
                    && otherCell->cellTypeData.constructor.isReady) {
                    return true;
                }
            }
            return false;
        };
        auto matchSecondChoiceFunc = [&](Cell* const& otherCell) {
            if (otherCell->livingState != LivingState_Ready) {
                return false;
            }
            if (!cudaSimulationParameters.transmitterEnergyDistributionSameCreature.value || otherCell->creatureId == cell->creatureId) {
                if (otherCell->cellType == CellType_Depot) {
                    return true;
                }
                if (otherCell->cellType == CellType_Constructor && !otherCell->cellTypeData.constructor.isReady) {
                    return true;
                }
            }
            return false;
        };

        Cell* receiverCells[20];
        int numReceivers;
        auto const& radius = cudaSimulationParameters.transmitterEnergyDistributionRadius.value[cell->color];
        data.cellMap.getMatchingCells(receiverCells, 20, numReceivers, cell->pos, radius, cell->detached, matchActiveConstructorFunc);
        if (numReceivers == 0) {
            data.cellMap.getMatchingCells(receiverCells, 20, numReceivers, cell->pos, radius, cell->detached, matchSecondChoiceFunc);
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
