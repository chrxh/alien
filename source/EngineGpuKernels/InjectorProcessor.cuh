#pragma once

#include "SignalProcessor.cuh"
#include "SimulationData.cuh"

class InjectorProcessor
{
public:
    __inline__ __device__ static void process(SimulationData& data, SimulationStatistics& statistics);

private:
    __inline__ __device__ static void processCell(SimulationData& data, SimulationStatistics& statistics, Object* cell);
    __inline__ __device__ static int countDefenderCells(SimulationStatistics& statistics, Object* cell);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__device__ __inline__ void InjectorProcessor::process(SimulationData& data, SimulationStatistics& statistics)
{
    auto& operations = data.cellTypeOperations[CellType_Injector];
    auto partition = calcSystemThreadPartition(operations.getNumEntries());
    for (int i = partition.startIndex; i <= partition.endIndex; i += partition.step) {
        processCell(data, statistics, operations.at(i).cell);
    }
}

__inline__ __device__ void InjectorProcessor::processCell(SimulationData& data, SimulationStatistics& statistics, Object* cell)
{
    if (SignalProcessor::isManuallyTriggered(data, cell)) {
        auto injectorEnergyCost = cudaSimulationParameters.injectorEnergyCost.value[object->color];
        auto cellMinEnergy = ParameterCalculator::calcParameter(cudaSimulationParameters.minCellEnergy, data, object->pos, object->color);

        Object* injectedCell = nullptr;
        int numDefenders = 0;
        data.cellMap.executeForEach(object->pos, cudaSimulationParameters.injectorRadius.value[object->color], object->detached, [&](auto const& otherCell) {
            if (injectedCell != nullptr) {
                return;
            }
            if (otherCell->creature == nullptr) {
                return;
            }
            if (object->isSameCreature(otherCell)) {
                return;
            }
            if (otherCell->fixed) {
                return;
            }
            if (otherCell->cellType != CellType_Constructor) {
                return;
            }
            // Only inject to other cells which are in a visible cone with respect to the injector cell
            if (ObjectConnectionProcessor::existsOwnIntersectingCellInBetween(data, cell, otherCell)) {
                return;
            }
            injectedCell = otherCell;
            numDefenders = countDefenderCells(statistics, otherCell);
        });

        if (injectedCell) {
            injectorEnergyCost *= (1.0f + toFloat(numDefenders) * 0.5f);
            if (object->usableEnergy - injectorEnergyCost < cellMinEnergy) {
                object->signal.channels[Channels::InjectorSuccess] = 0;
                return;
            }

            EntityFactory factory;
            factory.init(&data);
            auto cloneCreature = factory.cloneCreature(object->creature);
            cloneCreature->numObjects = 1;
            injectedCell->creature = cloneCreature;
            injectedCell->cellTypeData.constructor.geneIndex = object->cellTypeData.injector.geneIndex;
            injectedCell->cellTypeData.constructor.currentNodeIndex = 0;
            injectedCell->cellTypeData.constructor.currentConcatenation = 0;
            injectedCell->cellTypeData.constructor.currentBranch = 0;
            injectedCell->cellTypeData.constructor.lastConstructedCellId = VALUE_NOT_SET_UINT64;
            object->signal.channels[Channels::InjectorSuccess] = 1;

            if (injectorEnergyCost > 0) {
                EnergyParticleProcessor::radiate(data, cell, injectorEnergyCost);
            }
        }

        object->signal.channels[Channels::InjectorSuccess] = injectedCell != nullptr ? 1 : 0;
    }
}

__inline__ __device__ int InjectorProcessor::countDefenderCells(SimulationStatistics& statistics, Object* cell)
{
    int result = 0;
    for (int i = 0; i < object->numConnections; ++i) {
        auto connectedCell = object->connections[i].cell;
        if (connectedCell->cellType == CellType_Defender && connectedCell->cellTypeData.defender.mode == DefenderMode_DefendAgainstInjector) {
            statistics.incNumDefenderActivities(connectedCell->color);
            ++result;
        }
    }
    return result;
}
