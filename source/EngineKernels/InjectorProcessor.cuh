#pragma once

#include "SimulationData.cuh"

class InjectorProcessor
{
public:
    __inline__ __device__ static void process(SimulationData& data, SimulationStatistics& statistics);

private:
    __inline__ __device__ static void processCell(SimulationData& data, SimulationStatistics& statistics, Object* object);
    __inline__ __device__ static int countDefenderCells(SimulationStatistics& statistics, Object* object);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__device__ __inline__ void InjectorProcessor::process(SimulationData& data, SimulationStatistics& statistics)
{
    auto& operations = data.cellTypeOperations[CellType_Injector];
    auto partition = calcSystemThreadPartition(operations.getNumEntries());
    for (int i = partition.startIndex; i <= partition.endIndex; i += partition.step) {
        processCell(data, statistics, operations.at(i).object);
    }
}

__inline__ __device__ void InjectorProcessor::processCell(SimulationData& data, SimulationStatistics& statistics, Object* object)
{
    if (NeuronProcessor::isManuallyTriggered(data, object)) {
        auto injectorEnergyCost = cudaSimulationParameters.injectorEnergyCost.value[object->color];
        auto cellMinEnergy = ParameterCalculator::calcParameter(cudaSimulationParameters.minCellEnergy, data, object->pos, object->color);

        Object* injectedCell = nullptr;
        int numDefenders = 0;
        data.objectMap.executeForEach(
            object->pos, cudaSimulationParameters.injectorRadius.value[object->color], object->detached, [&](auto const& otherObject) {
                if (injectedCell != nullptr) {
                    return;
                }
                if (otherObject->type != ObjectType_Cell) {
                    return;
                }
                if (object->typeData.cell.isSameCreature(&otherObject->typeData.cell)) {
                    return;
                }
                if (otherObject->fixed) {
                    return;
                }
                if (!otherObject->typeData.cell.constructorAvailable) {
                    return;
                }
                // Only inject to other cells which are in a visible cone with respect to the injector cell
                if (ObjectConnectionProcessor::existsOwnIntersectingObjectInBetween(data, object, otherObject)) {
                    return;
                }
                injectedCell = otherObject;
                numDefenders = countDefenderCells(statistics, otherObject);
            });

        if (injectedCell) {
            injectorEnergyCost *= (1.0f + toFloat(numDefenders) * 0.5f);
            if (object->typeData.cell.usableEnergy - injectorEnergyCost < cellMinEnergy) {
                object->typeData.cell.signal.channels[Channels::InjectorSuccess] = 0;
                return;
            }

            EntityFactory factory;
            factory.init(&data);
            auto cloneCreature = factory.cloneCreature(object->typeData.cell.creature);
            cloneCreature->numCells = 1;
            injectedCell->typeData.cell.creature = cloneCreature;
            injectedCell->typeData.cell.constructor.geneIndex = object->typeData.cell.cellTypeData.injector.geneIndex;
            injectedCell->typeData.cell.constructor.lastConstructedCellId = VALUE_NOT_SET_UINT64;
            object->typeData.cell.signal.channels[Channels::InjectorSuccess] = 1;

            if (injectorEnergyCost > 0) {
                EnergyProcessor::radiate(data, object, injectorEnergyCost);
            }
        }

        object->typeData.cell.signal.channels[Channels::InjectorSuccess] = injectedCell != nullptr ? 1 : 0;
    }
}

__inline__ __device__ int InjectorProcessor::countDefenderCells(SimulationStatistics& statistics, Object* object)
{
    int result = 0;
    for (int i = 0; i < object->numConnections; ++i) {
        auto connectedObject = object->connections[i].object;
        if (connectedObject->typeData.cell.cellType == CellType_Defender
            && connectedObject->typeData.cell.cellTypeData.defender.mode == DefenderMode_DefendAgainstInjector) {
            statistics.incNumDefenderActivities(connectedObject->color);
            ++result;
        }
    }
    return result;
}
