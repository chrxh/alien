#pragma once

#include "EngineInterface/CellFunctionConstants.h"

#include "Cell.cuh"
#include "CellFunctionProcessor.cuh"
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
    auto& operations = data.cellFunctionOperations[CellFunction_Transmitter];
    auto partition = calcAllThreadsPartition(operations.getNumEntries());
    for (int i = partition.startIndex; i <= partition.endIndex; ++i) {
        auto const& cell = operations.at(i).cell;
        processCell(data, statistics, cell);
    }
}

__device__ __inline__ void TransmitterProcessor::processCell(SimulationData& data, SimulationStatistics& statistics, Cell* cell)
{
    auto activity = CellFunctionProcessor::calcInputActivity(cell);

    distributeEnergy(data, statistics, cell);

    CellFunctionProcessor::setActivity(cell, activity);
}

__device__ __inline__ void TransmitterProcessor::distributeEnergy(SimulationData& data, SimulationStatistics& statistics, Cell* cell)
{
    float energyDelta = 0;
    auto const& energyDistribution = cudaSimulationParameters.cellFunctionTransmitterEnergyDistributionValue[cell->color];
    auto origEnergy = atomicAdd(&cell->energy, -energyDistribution);
    if (origEnergy > cudaSimulationParameters.cellNormalEnergy[cell->color]) {
        energyDelta = energyDistribution;
    } else {
        atomicAdd(&cell->energy, energyDistribution);  //revert
    }
    for (int i = 0; i < cell->numConnections; ++i) {
        auto connectedCell = cell->connections[i].cell;
        if (connectedCell->cellFunction == CellFunction_Transmitter) {
            continue;
        }
        if (connectedCell->cellFunction == CellFunction_Constructor) {
            auto constructor = connectedCell->cellFunctionData.constructor;
            if (!GenomeDecoder::isFinishedSingleConstruction(constructor)) {
                continue;
            }
        }
        auto origEnergy = atomicAdd(&connectedCell->energy, -energyDistribution);
        if (origEnergy > cudaSimulationParameters.cellNormalEnergy[cell->color]) {
            energyDelta += energyDistribution;
        } else {
            atomicAdd(&connectedCell->energy, energyDistribution);  //revert
        }
    }

    if (energyDelta > NEAR_ZERO) {
        statistics.incNumTransmitterActivities(cell->color);
    }

    if (cell->cellFunctionData.transmitter.mode == EnergyDistributionMode_ConnectedCells) {
        int numReceivers = cell->numConnections;
        for (int i = 0; i < cell->numConnections; ++i) {
            numReceivers += cell->connections[i].cell->numConnections;
        }
        float energyPerReceiver = energyDelta / (numReceivers + 1);

        for (int i = 0; i < cell->numConnections; ++i) {
            auto connectedCell = cell->connections[i].cell;
            if (cudaSimulationParameters.cellFunctionTransmitterEnergyDistributionSameColor && connectedCell->color != cell->color) {
                continue;
            }
            atomicAdd(&connectedCell->energy, energyPerReceiver);

            energyDelta -= energyPerReceiver;
            for (int i = 0; i < connectedCell->numConnections; ++i) {
                auto connectedConnectedCell = connectedCell->connections[i].cell;
                if (cudaSimulationParameters.cellFunctionTransmitterEnergyDistributionSameColor && connectedConnectedCell->color != cell->color) {
                    continue;
                }
                atomicAdd(&connectedConnectedCell->energy, energyPerReceiver);
                energyDelta -= energyPerReceiver;
            }
        }
    }

    if (cell->cellFunctionData.transmitter.mode == EnergyDistributionMode_TransmittersAndConstructors) {
        auto matchActiveConstructorFunc = [&](Cell* const& otherCell) {
            if (otherCell->livingState != LivingState_Ready) {
                return false;
            }
            if (otherCell->cellFunction == CellFunction_Constructor) {
                auto isFinished = GenomeDecoder::isFinishedSingleConstruction(otherCell->cellFunctionData.constructor);
                if (!isFinished && (!cudaSimulationParameters.cellFunctionTransmitterEnergyDistributionSameColor || otherCell->color == cell->color)) {
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
                if (!cudaSimulationParameters.cellFunctionTransmitterEnergyDistributionSameColor || otherCell->color == cell->color) {
                    return true;
                }
            }
            return false;
        };

        Cell* receiverCells[20];
        int numReceivers;
        auto const& radius = cudaSimulationParameters.cellFunctionTransmitterEnergyDistributionRadius[cell->color];
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
