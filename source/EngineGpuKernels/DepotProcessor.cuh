#pragma once

#include <EngineInterface/CellTypeConstants.h>

#include "ConstantMemory.cuh"
#include "ConstructorHelper.cuh"
#include "Object.cuh"
#include "SignalProcessor.cuh"
#include "SimulationData.cuh"
#include "SimulationStatistics.cuh"

class DepotProcessor
{
public:
    __inline__ __device__ static void process(SimulationData& data, SimulationStatistics& statistics);

private:
    __inline__ __device__ static void processCell(SimulationData& data, SimulationStatistics& statistics, Cell* cell);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__device__ __inline__ void DepotProcessor::process(SimulationData& data, SimulationStatistics& statistics)
{
    auto& operations = data.cellTypeOperations[CellType_Depot];
    auto partition = calcSystemThreadPartitionNew(operations.getNumEntries());
    for (int i = partition.startIndex; i <= partition.endIndex; i += partition.step) {
        auto const& cell = operations.at(i).cell;
        processCell(data, statistics, cell);
    }
}

__device__ __inline__ void DepotProcessor::processCell(SimulationData& data, SimulationStatistics& statistics, Cell* cell)
{
    if (SignalProcessor::isManuallyTriggered(data, cell)) {
        auto normalCellEnergy = cudaSimulationParameters.normalCellEnergy.value[cell->color];
        if (cell->signal.channels[Channels::CellTypeActivation] > 0 && cell->usableEnergy > normalCellEnergy) {
            auto energyToTransfer = max(min(cell->usableEnergy - normalCellEnergy, SimulationParameters::depotEnergyTransferUnit), 0.0f);
            auto storageLimit = min(cell->cellTypeData.depot.storageLimit, SimulationParameters::depotStorageLimit);
            if (cell->cellTypeData.depot.storedUsableEnergy + energyToTransfer > storageLimit) {
                energyToTransfer = storageLimit - cell->cellTypeData.depot.storedUsableEnergy;
            }
            cell->usableEnergy -= energyToTransfer;
            cell->cellTypeData.depot.storedUsableEnergy += energyToTransfer;
        }
        if (cell->signal.channels[Channels::CellTypeActivation] < 0 && cell->cellTypeData.depot.storedUsableEnergy > 0) {
            auto energyToTransfer = max(min(cell->cellTypeData.depot.storedUsableEnergy, SimulationParameters::depotEnergyTransferUnit), 0.0f);
            cell->usableEnergy += energyToTransfer;
            cell->cellTypeData.depot.storedUsableEnergy -= energyToTransfer;
        }
    }
}