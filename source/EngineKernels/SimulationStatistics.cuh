#pragma once

#include <EngineInterface/StatisticsEntry.h>

#include "Base.cuh"
#include "LineageIdMap.cuh"

// Collects statistics of a single timestep (object counters and per-lineage values) plus accumulated per-lineage
// values that survive across timesteps. Instances are copied by value into kernels; all state lives in device memory.
class SimulationStatistics
{
    struct LineageMapEntry;  // Defined below

public:
    using LineageMap = LineageIdMap<LineageMapEntry>;

    static auto constexpr LineageMapCapacity = LineageMap::Capacity;
    static auto constexpr NoLineageSlot = LineageMap::NoLineageSlot;

    __host__ void init();
    __host__ void free();
    __host__ StatisticsEntry getStatisticsEntry() const;
    __host__ bool isLineageAccumulatorGCNeeded() const;

    // Object statistics (recalculated in every statistics timestep)
    __inline__ __device__ void resetObjectStatistics();
    __inline__ __device__ void incNumSolidObjects();
    __inline__ __device__ void incNumFluidObjects();
    __inline__ __device__ void incNumFreeCellObjects();
    __inline__ __device__ void incNumCellObjects();
    __inline__ __device__ void incNumEnergyParticles();
    __inline__ __device__ void addInternalEnergy(float value);

    // Lineage statistics (recalculated in every statistics timestep)
    __inline__ __device__ void resetLineageMapSlot(int index);
    __inline__ __device__ void resetLineageMapCounters();
    __inline__ __device__ int insertOrFindLineageSlot(uint32_t lineageId);
    __inline__ __device__ void addLineageCreatureData(int slotIndex, uint32_t numCells, uint32_t generation);
    __inline__ __device__ void addLineageGenomeData(int slotIndex, uint32_t numNodes, float meanMutationRate, uint32_t nodeColorBitset);
    __inline__ __device__ void addLineageEnergy(int slotIndex, float energy);
    __inline__ __device__ void updateLineageRepresentativeCell(int slotIndex, uint32_t generation, uint64_t cellId);

    // Copies the used slots of the lineage map into a dense array which can be read back by the host
    __inline__ __device__ void compactLineageSlot(int index);

    // Accumulated lineage statistics (kept across timesteps)
    __inline__ __device__ void incCreatedCreature(uint32_t lineageId);
    __inline__ __device__ void addMutations(uint32_t lineageId, float value);

    // Garbage collection: the accumulator map fills up over time because extinct lineages are never removed from it.
    // Therefore all lineages that are still alive are rebuilt into the inactive map, which then becomes the active one.
    __inline__ __device__ void resetInactiveAccumulatorMapCounters();
    __inline__ __device__ void resetInactiveAccumulatorSlot(int index);
    __inline__ __device__ void migrateActiveAccumulatorSlot(int index);
    __inline__ __device__ void flipAccumulatorMaps();

private:
    struct LineageMapEntry
    {
        uint32_t lineageId;  // Key, EmptyLineageId = slot is unused
        uint32_t colorBitset;
        uint32_t numCreatures;
        uint32_t numGenomes;
        uint64_t sumCreatureCells;
        uint64_t sumCreatureGenerations;
        uint64_t sumGenomeNodes;
        double sumMutationRates;
        double sumCreatureEnergy;
        uint64_t representativeCellId;   // Cell of a creature with the highest generation; 0 = not set
        uint32_t maxCreatureGeneration;  // Only needed to determine the representative cell

        __inline__ __device__ LineageStatisticsEntry toStatisticsEntry() const;
    };
    struct LineageAccumulatorEntry
    {
        uint32_t lineageId;  // Key, EmptyLineageId = slot is unused
        uint64_t numCreatedCreatures;
        double totalMutations;
    };
    struct StatisticsControl
    {
        uint32_t numCompactedLineageEntries;  // Number of valid elements in _lineageStatisticsEntries
        uint32_t activeAccumulatorMapIndex;
    };

    using AccumulatorMap = LineageIdMap<LineageAccumulatorEntry>;

    __host__ StatisticsControl readControl() const;

    __inline__ __device__ AccumulatorMap& getActiveAccumulatorMap();
    __inline__ __device__ AccumulatorMap const& getActiveAccumulatorMap() const;
    __inline__ __device__ AccumulatorMap& getInactiveAccumulatorMap();

    static auto constexpr AccumulatorGCLoadFactor = 0.5f;

    StatisticsControl* _control;

    ObjectStatisticsEntry* _objectStatisticsEntry;

    // Values of the current timestep and their dense representation for the host
    LineageMap _lineageMap;
    LineageStatisticsEntry* _lineageStatisticsEntries;

    // Accumulated values, double buffered for garbage collection
    AccumulatorMap _accumulatorMaps[2];
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__inline__ __device__ void SimulationStatistics::resetObjectStatistics()
{
    *_objectStatisticsEntry = ObjectStatisticsEntry{};
}

__inline__ __device__ void SimulationStatistics::incNumSolidObjects()
{
    atomicAdd(&_objectStatisticsEntry->numSolidObjects, 1u);
}

__inline__ __device__ void SimulationStatistics::incNumFluidObjects()
{
    atomicAdd(&_objectStatisticsEntry->numFluidObjects, 1u);
}

__inline__ __device__ void SimulationStatistics::incNumFreeCellObjects()
{
    atomicAdd(&_objectStatisticsEntry->numFreeCellObjects, 1u);
}

__inline__ __device__ void SimulationStatistics::incNumCellObjects()
{
    atomicAdd(&_objectStatisticsEntry->numCellObjects, 1u);
}

__inline__ __device__ void SimulationStatistics::incNumEnergyParticles()
{
    atomicAdd(&_objectStatisticsEntry->numEnergyParticles, 1u);
}

__inline__ __device__ void SimulationStatistics::addInternalEnergy(float value)
{
    atomicAdd(&_objectStatisticsEntry->totalInternalEnergy, toDouble(value));
}

__inline__ __device__ void SimulationStatistics::resetLineageMapSlot(int index)
{
    _lineageMap.resetSlot(index);
}

__inline__ __device__ void SimulationStatistics::resetLineageMapCounters()
{
    _lineageMap.resetNumUsedSlots();
    _control->numCompactedLineageEntries = 0;
}

__inline__ __device__ int SimulationStatistics::insertOrFindLineageSlot(uint32_t lineageId)
{
    return _lineageMap.insertOrFind(lineageId);
}

__inline__ __device__ void SimulationStatistics::addLineageCreatureData(int slotIndex, uint32_t numCells, uint32_t generation)
{
    auto& slot = _lineageMap.at(slotIndex);
    atomicAdd(&slot.numCreatures, 1u);
    alienAtomicAdd64(&slot.sumCreatureCells, static_cast<uint64_t>(numCells));
    alienAtomicAdd64(&slot.sumCreatureGenerations, static_cast<uint64_t>(generation));
    atomicMax(&slot.maxCreatureGeneration, generation);
}

__inline__ __device__ void SimulationStatistics::addLineageGenomeData(int slotIndex, uint32_t numNodes, float meanMutationRate, uint32_t nodeColorBitset)
{
    auto& slot = _lineageMap.at(slotIndex);
    atomicAdd(&slot.numGenomes, 1u);
    alienAtomicAdd64(&slot.sumGenomeNodes, static_cast<uint64_t>(numNodes));
    atomicAdd(&slot.sumMutationRates, toDouble(meanMutationRate));
    atomicOr(&slot.colorBitset, nodeColorBitset);
}

__inline__ __device__ void SimulationStatistics::addLineageEnergy(int slotIndex, float energy)
{
    atomicAdd(&_lineageMap.at(slotIndex).sumCreatureEnergy, toDouble(energy));
}

__inline__ __device__ void SimulationStatistics::updateLineageRepresentativeCell(int slotIndex, uint32_t generation, uint64_t cellId)
{
    auto& slot = _lineageMap.at(slotIndex);
    if (generation == slot.maxCreatureGeneration) {
        atomicCAS(reinterpret_cast<unsigned long long*>(&slot.representativeCellId), 0ull, static_cast<unsigned long long>(cellId));
    }
}

__inline__ __device__ void SimulationStatistics::compactLineageSlot(int index)
{
    auto const& slot = _lineageMap.at(index);
    if (slot.lineageId == LineageMap::EmptyLineageId || slot.numCreatures == 0) {
        return;
    }
    auto entryIndex = atomicAdd(&_control->numCompactedLineageEntries, 1u);
    auto& entry = _lineageStatisticsEntries[entryIndex];
    entry = slot.toStatisticsEntry();

    auto const& accumulatorMap = getActiveAccumulatorMap();
    auto accumulatorIndex = accumulatorMap.find(slot.lineageId);
    if (accumulatorIndex != NoLineageSlot) {
        auto const& accumulatorSlot = accumulatorMap.at(accumulatorIndex);
        entry.numCreatedCreatures = accumulatorSlot.numCreatedCreatures;
        entry.totalMutations = accumulatorSlot.totalMutations;
    }
}

__inline__ __device__ void SimulationStatistics::incCreatedCreature(uint32_t lineageId)
{
    auto& map = getActiveAccumulatorMap();
    auto slotIndex = map.insertOrFind(lineageId);
    if (slotIndex != NoLineageSlot) {
        alienAtomicAdd64(&map.at(slotIndex).numCreatedCreatures, static_cast<uint64_t>(1));
    }
}

__inline__ __device__ void SimulationStatistics::addMutations(uint32_t lineageId, float value)
{
    auto& map = getActiveAccumulatorMap();
    auto slotIndex = map.insertOrFind(lineageId);
    if (slotIndex != NoLineageSlot) {
        atomicAdd(&map.at(slotIndex).totalMutations, value);
    }
}

__inline__ __device__ void SimulationStatistics::resetInactiveAccumulatorMapCounters()
{
    getInactiveAccumulatorMap().resetNumUsedSlots();
}

__inline__ __device__ void SimulationStatistics::resetInactiveAccumulatorSlot(int index)
{
    getInactiveAccumulatorMap().resetSlot(index);
}

__inline__ __device__ void SimulationStatistics::migrateActiveAccumulatorSlot(int index)
{
    auto const& slot = getActiveAccumulatorMap().at(index);
    if (slot.lineageId == LineageMap::EmptyLineageId || _lineageMap.find(slot.lineageId) == NoLineageSlot) {
        return;
    }
    auto& targetMap = getInactiveAccumulatorMap();
    auto targetIndex = targetMap.insertOrFind(slot.lineageId);
    if (targetIndex != NoLineageSlot) {
        auto& targetSlot = targetMap.at(targetIndex);
        targetSlot.numCreatedCreatures = slot.numCreatedCreatures;
        targetSlot.totalMutations = slot.totalMutations;
    }
}

__inline__ __device__ void SimulationStatistics::flipAccumulatorMaps()
{
    _control->activeAccumulatorMapIndex = 1 - _control->activeAccumulatorMapIndex;
}

__inline__ __device__ SimulationStatistics::AccumulatorMap& SimulationStatistics::getActiveAccumulatorMap()
{
    return _accumulatorMaps[_control->activeAccumulatorMapIndex];
}

__inline__ __device__ SimulationStatistics::AccumulatorMap const& SimulationStatistics::getActiveAccumulatorMap() const
{
    return _accumulatorMaps[_control->activeAccumulatorMapIndex];
}

__inline__ __device__ SimulationStatistics::AccumulatorMap& SimulationStatistics::getInactiveAccumulatorMap()
{
    return _accumulatorMaps[1 - _control->activeAccumulatorMapIndex];
}

__inline__ __device__ LineageStatisticsEntry SimulationStatistics::LineageMapEntry::toStatisticsEntry() const
{
    LineageStatisticsEntry result;
    result.lineageId = lineageId;
    result.colorBitset = colorBitset;
    result.numCreatures = numCreatures;
    result.numGenomes = numGenomes;
    result.sumCreatureCells = sumCreatureCells;
    result.sumCreatureGenerations = sumCreatureGenerations;
    result.sumGenomeNodes = sumGenomeNodes;
    result.sumMutationRates = sumMutationRates;
    result.sumCreatureEnergy = sumCreatureEnergy;
    result.representativeCellId = representativeCellId;
    return result;  // The accumulated members are taken from the accumulator map
}
