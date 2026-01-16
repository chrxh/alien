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
    __inline__ __device__ static void processCell(SimulationData& data, SimulationStatistics& statistics, Object* object);
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
        processCell(data, statistics, object);
    }
}

__device__ __inline__ void DepotProcessor::processCell(SimulationData& data, SimulationStatistics& statistics, Object* object)
{
    if (SignalProcessor::isManuallyTriggered(data, object)) {
        auto normalCellEnergy = cudaSimulationParameters.normalCellEnergy.value[object->color];
        if (object->typeData.cell.signal.channels[Channels::CellTypeActivation] > 0 && object->typeData.cell.usableEnergy > normalCellEnergy) {
            auto energyToTransfer = max(min(object->typeData.cell.usableEnergy - normalCellEnergy, SimulationParameters::depotEnergyTransferUnit), 0.0f);
            auto storageLimit = min(object->typeData.cell.cellTypeData.depot.storageLimit, SimulationParameters::depotStorageLimit);
            if (object->typeData.cell.cellTypeData.depot.storedUsableEnergy + energyToTransfer > storageLimit) {
                energyToTransfer = storageLimit - object->typeData.cell.cellTypeData.depot.storedUsableEnergy;
            }
            object->typeData.cell.usableEnergy -= energyToTransfer;
            object->typeData.cell.cellTypeData.depot.storedUsableEnergy += energyToTransfer;
        }
        if (object->typeData.cell.signal.channels[Channels::CellTypeActivation] < 0 && object->typeData.cell.cellTypeData.depot.storedUsableEnergy > 0) {
            auto energyToTransfer = max(min(object->typeData.cell.cellTypeData.depot.storedUsableEnergy, SimulationParameters::depotEnergyTransferUnit), 0.0f);
            object->typeData.cell.usableEnergy += energyToTransfer;
            object->typeData.cell.cellTypeData.depot.storedUsableEnergy -= energyToTransfer;
        }
    }
}