#include "StatisticsKernels.cuh"

namespace
{
    __device__ float calcMeanMutationRate(MutationRates const& rates)
    {
        auto sum = 0.0f;
        for (int i = 0; i < 2; ++i) {
            sum += rates.neuronMutations[i].nodeProbability;
            sum += rates.connectionMutations[i].nodeProbability;
            sum += rates.cellTypePropertiesMutations[i].nodeProbability;
            sum += rates.geometryMutations[i].geneProbability;
            sum += rates.constructorMutations[i].nodeProbability;
        }
        sum += rates.cellTypeModeMutation.nodeProbability;
        sum += rates.cellTypeMutation.nodeProbability;
        sum += rates.voidMutation.nodeProbability;
        sum += rates.extendGeneMutation.geneProbability;
        sum += rates.addNodeMutation.nodeProbability;
        sum += rates.trimGeneMutation.geneProbability;
        sum += rates.deleteNodeMutation.nodeProbability;
        sum += rates.copyNodeSectionMutation.geneProbability;
        sum += rates.moveNodeSectionMutation.geneProbability;
        sum += rates.duplicateGeneMutation.geneProbability;
        sum += rates.deleteGeneMutation.geneProbability;
        return sum / 21.0f;
    }

    __device__ int getCachedLineageSlot(Creature* creature, int lineageMapCapacity)
    {
        auto slotIndexPlusOne = creature->creatureIndex;
        if (slotIndexPlusOne >= 1 && slotIndexPlusOne <= static_cast<uint64_t>(lineageMapCapacity)) {
            return toInt(slotIndexPlusOne) - 1;
        }
        return -1;
    }
}

__global__ void cudaUpdateEvolutionStatistics_substep1(SimulationData data, SimulationStatistics statistics)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        statistics.resetEvolutionStatistics();
        statistics.resetCompactLineageCounter();
    }
    {
        auto const partition = calcSystemThreadPartition(statistics.getLineageMapCapacity());
        for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
            statistics.resetLineageMapSlot(index);
        }
    }
    {
        auto& objects = data.entities.objects;
        auto const partition = calcSystemThreadPartition(objects.getNumEntries());
        for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
            auto& object = objects.at(index);
            if (object->type != ObjectType_Cell) {
                continue;
            }
            auto creature = object->typeData.cell.creature;
            if (!creature) {
                continue;
            }
            creature->creatureIndex = VALUE_NOT_SET_UINT64;
            creature->genome->genomeIndex = VALUE_NOT_SET_UINT64;
        }
    }
}

__global__ void cudaUpdateEvolutionStatistics_substep2(SimulationData data, SimulationStatistics statistics)
{
    {
        auto& particles = data.entities.energies;
        auto const partition = calcSystemThreadPartition(particles.getNumEntries());
        for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
            if (particles.at(index)) {
                statistics.incNumEnergyParticles();
            }
        }
    }
    auto& objects = data.entities.objects;
    auto const partition = calcSystemThreadPartition(objects.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = objects.at(index);
        if (object->type == ObjectType_Solid) {
            statistics.incNumSolidObjects();
        } else if (object->type == ObjectType_Fluid) {
            statistics.incNumFluidObjects();
        } else if (object->type == ObjectType_Cell) {
            statistics.incNumCellObjects();
        }
        if (object->type != ObjectType_Cell) {
            continue;
        }
        auto creature = object->typeData.cell.creature;
        if (!creature) {
            continue;
        }
        auto origCreatureIndex = alienAtomicExch64(&creature->creatureIndex, static_cast<uint64_t>(0));  // 0 = member is currently initialized
        if (origCreatureIndex == VALUE_NOT_SET_UINT64) {
            statistics.addCreatureStatistics(creature->numCells, creature->generation, creature->accumulatedMutations);
            auto slotIndex = statistics.insertOrFindLineageSlot(creature->lineageId);
            if (slotIndex >= 0) {
                statistics.addLineageCreatureData(slotIndex, creature->numCells, creature->generation);
                alienAtomicExch64(&creature->creatureIndex, static_cast<uint64_t>(slotIndex) + 1);
            }
        } else if (origCreatureIndex != 0) {
            // Another thread already finished the initialization; restore its value (see DataAccessKernels)
            alienAtomicExch64(&creature->creatureIndex, origCreatureIndex);
        }
    }
}

__global__ void cudaUpdateEvolutionStatistics_substep3(SimulationData data, SimulationStatistics statistics)
{
    auto& objects = data.entities.objects;
    auto const partition = calcSystemThreadPartition(objects.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = objects.at(index);
        if (object->type != ObjectType_Cell) {
            continue;
        }
        auto creature = object->typeData.cell.creature;
        if (!creature) {
            continue;
        }
        auto slotIndex = getCachedLineageSlot(creature, statistics.getLineageMapCapacity());

        auto genome = creature->genome;
        auto origGenomeIndex = alienAtomicExch64(&genome->genomeIndex, static_cast<uint64_t>(0));
        if (origGenomeIndex == VALUE_NOT_SET_UINT64) {
            auto numNodes = 0;
            auto nodeColorBitset = 0u;
            for (int i = 0; i < genome->numGenes; ++i) {
                auto const& gene = genome->genes[i];
                numNodes += gene.numNodes;
                for (int j = 0; j < gene.numNodes; ++j) {
                    nodeColorBitset |= 1u << gene.nodes[j].color;
                }
            }
            auto meanMutationRate = calcMeanMutationRate(genome->mutationRates);
            statistics.addGenomeStatistics(toFloat(numNodes), meanMutationRate);
            if (slotIndex >= 0) {
                statistics.addLineageGenomeData(slotIndex, toFloat(numNodes), meanMutationRate, nodeColorBitset);
            }
        }

        auto energy = object->getEnergy();
        statistics.addCreatureEnergy(energy);
        if (slotIndex >= 0) {
            statistics.addLineageEnergy(slotIndex, energy);
            statistics.updateLineageRepresentativeCell(slotIndex, creature->generation, object->id);
        }
    }
}

__global__ void cudaUpdateEvolutionStatistics_substep4(SimulationData data, SimulationStatistics statistics)
{
    auto const partition = calcSystemThreadPartition(statistics.getLineageMapCapacity());
    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        statistics.compactLineageSlot(index);
    }
}

__global__ void cudaUpdateEvolutionStatistics_substep5(SimulationData data, SimulationStatistics statistics)
{
    statistics.finalizeLineageStatistics();
}

__global__ void cudaPrepareLineageAccumulatorGC(SimulationStatistics statistics)
{
    auto const partition = calcSystemThreadPartition(statistics.getLineageMapCapacity());
    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        statistics.resetInactiveAccumulatorSlot(index);
    }
}

__global__ void cudaLineageAccumulatorGC(SimulationStatistics statistics)
{
    auto const partition = calcSystemThreadPartition(statistics.getLineageMapCapacity());
    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        statistics.migrateActiveAccumulatorSlot(index);
    }
}

__global__ void cudaFinishLineageAccumulatorGC(SimulationStatistics statistics)
{
    statistics.flipAccumulatorBuffers();
}

