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
    if (cell->energy <= cudaSimulationParameters.cellNormalEnergy) {
        return;
    }
    if (!cell->tryLock()) {
        return;
    }

    float energyDelta = cell->energy - cudaSimulationParameters.cellNormalEnergy;
    cell->energy = cudaSimulationParameters.cellNormalEnergy;

    if (cell->cellFunctionData.transmitter.mode == Enums::EnergyDistributionMode_ConnectedCells) {
        int numReceivers = cell->numConnections;
        for (int i = 0; i < cell->numConnections; ++i) {
            numReceivers += cell->connections[i].cell->numConnections;
        }
        float energyPerReceiver = energyDelta / (numReceivers + 1);

        for (int i = 0; i < cell->numConnections; ++i) {
            auto connectedCell = cell->connections[i].cell;
            if (connectedCell->tryLock()) {
                connectedCell->energy += energyPerReceiver;
                energyDelta -= energyPerReceiver;
                connectedCell->releaseLock();
            }
            for (int i = 0; i < connectedCell->numConnections; ++i) {
                auto connectedConnectedCell = connectedCell->connections[i].cell;
                if (connectedConnectedCell->tryLock()) {
                    connectedConnectedCell->energy += energyPerReceiver;
                    energyDelta -= energyPerReceiver;
                    connectedConnectedCell->releaseLock();
                }
            }
        }
    }

    if (cell->cellFunctionData.attacker.mode == Enums::EnergyDistributionMode_TransmittersAndConstructors) {
        Cell* receiverCells[10];
        int numReceivers;
        data.cellMap.getActiveConstructors(
            receiverCells, 10, numReceivers, cell->absPos, cudaSimulationParameters.cellFunctionAttackerEnergyDistributionRadius);
        if (numReceivers == 0) {
            data.cellMap.getTransmitters(receiverCells, 10, numReceivers, cell->absPos, cudaSimulationParameters.cellFunctionAttackerEnergyDistributionRadius);
        }
        float energyPerReceiver = energyDelta / (numReceivers + 1);

        for (int i = 0; i < numReceivers; ++i) {
            auto receiverCell = receiverCells[i];
            if (receiverCell->tryLock()) {
                receiverCell->energy += energyPerReceiver;
                energyDelta -= energyPerReceiver;
                receiverCell->releaseLock();
            }
        }
    }
    cell->energy += energyDelta;

    cell->releaseLock();
}
