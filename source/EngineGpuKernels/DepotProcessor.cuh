#pragma once

#include <EngineInterface/CellTypeConstants.h>

#include "ConstantMemory.cuh"
#include "ConstructorHelper.cuh"
#include "Entity.cuh"
#include "SignalProcessor.cuh"
#include "SimulationData.cuh"
#include "SimulationStatistics.cuh"

class DepotProcessor
{
public:
    __inline__ __device__ static void process(SimulationData& data, SimulationStatistics& statistics);

private:
    __inline__ __device__ static void processCell(SimulationData& data, SimulationStatistics& statistics, Object* cell);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__device__ __inline__ void DepotProcessor::process(SimulationData& data, SimulationStatistics& statistics)
{
    auto& operations = data.cellTypeOperations[CellType_Depot];
    auto partition = calcSystemThreadPartition(operations.getNumEntries());
    for (int i = partition.startIndex; i <= partition.endIndex; i += partition.step) {
        auto const& object = operations.at(i).object;
        processCell(data, statistics, cell);
    }
}

__device__ __inline__ void DepotProcessor::processCell(SimulationData& data, SimulationStatistics& statistics, Object* cell)
{
    if (SignalProcessor::isManuallyTriggered(data, cell)) {
        auto normalCellEnergy = cudaSimulationParameters.normalCellEnergy.value[object->color];
        if (object->signal.channels[Channels::CellTypeActivation] > 0 && object->usableEnergy > normalCellEnergy) {
            auto energyToTransfer = max(min(object->usableEnergy - normalCellEnergy, SimulationParameters::depotEnergyTransferUnit), 0.0f);
            auto storageLimit = min(object->cellTypeData.depot.storageLimit, SimulationParameters::depotStorageLimit);
            if (object->cellTypeData.depot.storedUsableEnergy + energyToTransfer > storageLimit) {
                energyToTransfer = storageLimit - object->cellTypeData.depot.storedUsableEnergy;
            }
            object->usableEnergy -= energyToTransfer;
            object->cellTypeData.depot.storedUsableEnergy += energyToTransfer;
        }
        if (object->signal.channels[Channels::CellTypeActivation] < 0 && object->cellTypeData.depot.storedUsableEnergy > 0) {
            auto energyToTransfer = max(min(object->cellTypeData.depot.storedUsableEnergy, SimulationParameters::depotEnergyTransferUnit), 0.0f);
            object->usableEnergy += energyToTransfer;
            object->cellTypeData.depot.storedUsableEnergy -= energyToTransfer;
        }
    }
}