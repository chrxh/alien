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

    // The lineage slot of a creature is determined once per statistics timestep and cached in Creature::creatureIndex,
    // a scratch member that other kernels use for different purposes (see DataAccessKernels and EditKernels).
    // It is stored as slot index + 1, so that the sentinels 0 (currently initialized) and VALUE_NOT_SET_UINT64 keep their meaning.
    __device__ void cacheLineageSlot(Creature* creature, int slotIndex)
    {
        alienAtomicExch64(&creature->creatureIndex, static_cast<uint64_t>(slotIndex) + 1);
    }

    __device__ int getCachedLineageSlot(Creature* creature)
    {
        auto slotIndexPlusOne = creature->creatureIndex;
        if (slotIndexPlusOne >= 1 && slotIndexPlusOne <= static_cast<uint64_t>(SimulationStatistics::LineageMapCapacity)) {
            return toInt(slotIndexPlusOne) - 1;
        }
        return SimulationStatistics::NoLineageSlot;
    }
}

__global__ void cudaResetStatistics(SimulationData data, SimulationStatistics statistics)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        statistics.resetObjectStatistics();
        statistics.resetLineageMapCounters();
    }
    {
        auto const partition = calcSystemThreadPartition(SimulationStatistics::LineageMapCapacity);
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

__global__ void cudaCollectObjectAndCreatureStatistics(SimulationData data, SimulationStatistics statistics)
{
    {
        auto& particles = data.entities.energies;
        auto const partition = calcSystemThreadPartition(particles.getNumEntries());
        for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
            if (auto& particle = particles.at(index)) {
                statistics.incNumEnergyParticles();
                statistics.addInternalEnergy(particle->energy);
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
        } else if (object->type == ObjectType_FreeCell) {
            statistics.incNumFreeCellObjects();
        } else if (object->type == ObjectType_Cell) {
            statistics.incNumCellObjects();
        }
        statistics.addInternalEnergy(object->getEnergy());

        if (object->type != ObjectType_Cell) {
            continue;
        }
        auto creature = object->typeData.cell.creature;
        if (!creature) {
            continue;
        }
        auto origCreatureIndex = alienAtomicExch64(&creature->creatureIndex, static_cast<uint64_t>(0));  // 0 = member is currently initialized
        if (origCreatureIndex == VALUE_NOT_SET_UINT64) {
            auto slotIndex = statistics.insertOrFindLineageSlot(creature->lineageId);
            if (slotIndex != SimulationStatistics::NoLineageSlot) {
                statistics.addLineageCreatureData(slotIndex, creature->numCells, creature->generation);
                cacheLineageSlot(creature, slotIndex);
            }
        } else if (origCreatureIndex != 0) {
            // Another thread already finished the initialization; restore its value (see DataAccessKernels)
            alienAtomicExch64(&creature->creatureIndex, origCreatureIndex);
        }
    }
}

__global__ void cudaCollectGenomeAndEnergyStatistics(SimulationData data, SimulationStatistics statistics)
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
        auto slotIndex = getCachedLineageSlot(creature);

        auto genome = creature->genome;
        auto origGenomeIndex = alienAtomicExch64(&genome->genomeIndex, static_cast<uint64_t>(0));
        if (origGenomeIndex == VALUE_NOT_SET_UINT64 && slotIndex != SimulationStatistics::NoLineageSlot) {
            auto numNodes = 0u;
            auto nodeColorBitset = 0u;
            for (int i = 0; i < genome->numGenes; ++i) {
                auto const& gene = genome->genes[i];
                numNodes += gene.numNodes;
                for (int j = 0; j < gene.numNodes; ++j) {
                    nodeColorBitset |= 1u << gene.nodes[j].color;
                }
            }
            statistics.addLineageGenomeData(slotIndex, numNodes, calcMeanMutationRate(genome->mutationRates), nodeColorBitset);
        }

        if (slotIndex != SimulationStatistics::NoLineageSlot) {
            statistics.addLineageEnergy(slotIndex, object->getEnergy());
            statistics.updateLineageRepresentativeCell(slotIndex, creature->generation, object->id);
        }
    }
}

__global__ void cudaCompactLineageStatistics(SimulationStatistics statistics)
{
    auto const partition = calcSystemThreadPartition(SimulationStatistics::LineageMapCapacity);
    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        statistics.compactLineageSlot(index);
    }
}

__global__ void cudaPrepareLineageAccumulatorGC(SimulationStatistics statistics)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        statistics.resetInactiveAccumulatorMapCounters();
    }
    auto const partition = calcSystemThreadPartition(SimulationStatistics::LineageMapCapacity);
    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        statistics.resetInactiveAccumulatorSlot(index);
    }
}

__global__ void cudaLineageAccumulatorGC(SimulationStatistics statistics)
{
    auto const partition = calcSystemThreadPartition(SimulationStatistics::LineageMapCapacity);
    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        statistics.migrateActiveAccumulatorSlot(index);
    }
}

__global__ void cudaFinishLineageAccumulatorGC(SimulationStatistics statistics)
{
    statistics.flipAccumulatorMaps();
}
