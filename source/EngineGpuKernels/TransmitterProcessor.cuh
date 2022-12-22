#pragma once

#include "EngineInterface/Enums.h"

#include "Cell.cuh"
#include "CellFunctionProcessor.cuh"
#include "ConstantMemory.cuh"
#include "SimulationData.cuh"
#include "SimulationResult.cuh"

class TransmitterProcessor
{
public:
    __inline__ __device__ static void process(SimulationData& data, SimulationResult& result);

private:
    __inline__ __device__ static void processCell(SimulationData& data, SimulationResult& result, Cell* cell);
    __inline__ __device__ static void distributeEnergy(SimulationData& data, Cell* cell);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__device__ __inline__ void TransmitterProcessor::process(SimulationData& data, SimulationResult& result)
{
    auto& operations = data.cellFunctionOperations[Enums::CellFunction_Transmitter];
    auto partition = calcAllThreadsPartition(operations.getNumEntries());
    for (int i = partition.startIndex; i <= partition.endIndex; ++i) {
        auto const& cell = operations.at(i).cell;
        processCell(data, result, cell);
    }
}

__device__ __inline__ void TransmitterProcessor::processCell(SimulationData& data, SimulationResult& result, Cell* cell)
{
    int inputExecutionOrderNumber;
    auto activity = CellFunctionProcessor::calcInputActivity(cell, inputExecutionOrderNumber);

    distributeEnergy(data, cell);

    CellFunctionProcessor::setActivity(cell, activity);
}

__device__ __inline__ void TransmitterProcessor::distributeEnergy(SimulationData& data, Cell* cell)
{

    float energyDelta = 0;
    auto origEnergy = atomicAdd(&cell->energy, -cudaSimulationParameters.cellFunctionTransmitterEnergyDistributionValue);
    if (origEnergy > cudaSimulationParameters.cellNormalEnergy) {
        energyDelta = cudaSimulationParameters.cellFunctionTransmitterEnergyDistributionValue;
    } else {
        atomicAdd(&cell->energy, cudaSimulationParameters.cellFunctionTransmitterEnergyDistributionValue);  //revert
    }
    for (int i = 0; i < cell->numConnections; ++i) {
        auto connectedCell = cell->connections[i].cell;
        if (connectedCell->cellFunction == Enums::CellFunction_Constructor || connectedCell->cellFunction == Enums::CellFunction_Transmitter) {
            continue;
        }
        auto origEnergy = atomicAdd(&connectedCell->energy, -cudaSimulationParameters.cellFunctionTransmitterEnergyDistributionValue);
        if (origEnergy > cudaSimulationParameters.cellNormalEnergy) {
            energyDelta += cudaSimulationParameters.cellFunctionTransmitterEnergyDistributionValue;
        } else {
            atomicAdd(&connectedCell->energy, cudaSimulationParameters.cellFunctionTransmitterEnergyDistributionValue);  //revert
        }
    }

    if (cell->cellFunctionData.transmitter.mode == Enums::EnergyDistributionMode_ConnectedCells) {
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

    if (cell->cellFunctionData.attacker.mode == Enums::EnergyDistributionMode_TransmittersAndConstructors) {
        auto matchActiveConstructorFunc = [&](Cell* const& otherCell) {
            auto const& constructor = otherCell->cellFunctionData.constructor;
            auto const isActive = !(constructor.singleConstruction && constructor.currentGenomePos >= constructor.genomeSize);
            if (otherCell->cellFunction == Enums::CellFunction_Constructor && isActive) {
                if (!cudaSimulationParameters.cellFunctionAttackerEnergyDistributionSameColor || otherCell->color == cell->color) {
                    return true;
                }
            }
            return false;
        };
        auto matchTransmitterFunc = [&](Cell* const& otherCell) {
            if (otherCell->cellFunction == Enums::CellFunction_Transmitter) {
                if (!cudaSimulationParameters.cellFunctionAttackerEnergyDistributionSameColor || otherCell->color == cell->color) {
                    return true;
                }
            }
            return false;
        };

        Cell* receiverCells[10];
        int numReceivers;
        data.cellMap.getMatchingCells(
            receiverCells, 10, numReceivers, cell->absPos, cudaSimulationParameters.cellFunctionAttackerEnergyDistributionRadius, matchActiveConstructorFunc);
        if (numReceivers == 0) {
            data.cellMap.getMatchingCells(
                receiverCells, 10, numReceivers, cell->absPos, cudaSimulationParameters.cellFunctionAttackerEnergyDistributionRadius, matchTransmitterFunc);
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
