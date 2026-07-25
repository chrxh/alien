#pragma once

#include <algorithm>

#include <EngineInterface/StatisticsEntry.h>

#include "Base.cuh"
#include "CudaMemoryManager.cuh"

class SimulationStatistics
{
public:
    __host__ void init()
    {
        _lineageArrayCapacity = 1 << 18;

        CudaMemoryManager::getInstance().acquireMemory<ObjectStatisticsEntry>(1, _objectStatisticsEntry);
        CudaMemoryManager::getInstance().acquireMemory<LineageStatisticsEntry>(_lineageArrayCapacity, _lineageStatisticsEntries);

        CudaMemoryManager::getInstance().acquireMemory<LineageMapEntry>(_lineageArrayCapacity, _lineageMap);

        CudaMemoryManager::getInstance().acquireMemory<LineageAccumulatorMapControl>(1, _lineageAccumulatorMapControl);
        CudaMemoryManager::getInstance().acquireMemory<LineageAccumulatorMapEntry>(_lineageArrayCapacity, _lineageAccumulatorMaps[0]);
        CudaMemoryManager::getInstance().acquireMemory<LineageAccumulatorMapEntry>(_lineageArrayCapacity, _lineageAccumulatorMaps[1]);
        CHECK_FOR_DEVICE_ERRORS(cudaMemset(_objectStatisticsEntry, 0, sizeof(ObjectStatisticsEntry)));
        CHECK_FOR_DEVICE_ERRORS(cudaMemset(_lineageAccumulatorMapControl, 0, sizeof(LineageAccumulatorMapControl)));

        // Values must start at zero; the lineageId key column (first member) is set to LineageIdEmpty
        CHECK_FOR_DEVICE_ERRORS(cudaMemset(_lineageMap, 0, sizeof(LineageMapEntry) * _lineageArrayCapacity));
        CHECK_FOR_DEVICE_ERRORS(cudaMemset2D(_lineageMap, sizeof(LineageMapEntry), 0xff, sizeof(uint32_t), _lineageArrayCapacity));
        for (int i = 0; i < 2; ++i) {
            CHECK_FOR_DEVICE_ERRORS(cudaMemset(_lineageAccumulatorMaps[i], 0, sizeof(LineageAccumulatorMapEntry) * _lineageArrayCapacity));
            CHECK_FOR_DEVICE_ERRORS(cudaMemset2D(_lineageAccumulatorMaps[i], sizeof(LineageAccumulatorMapEntry), 0xff, sizeof(uint32_t), _lineageArrayCapacity));
        }
    }

    __host__ void free()
    {
        CudaMemoryManager::getInstance().freeMemory<ObjectStatisticsEntry>(_objectStatisticsEntry);
        CudaMemoryManager::getInstance().freeMemory(_lineageStatisticsEntries);

        CudaMemoryManager::getInstance().freeMemory(_lineageMap);

        CudaMemoryManager::getInstance().freeMemory(_lineageAccumulatorMapControl);
        CudaMemoryManager::getInstance().freeMemory(_lineageAccumulatorMaps[0]);
        CudaMemoryManager::getInstance().freeMemory(_lineageAccumulatorMaps[1]);
    }

    __host__ StatisticsEntry getStatisticsEntry()
    {
        StatisticsEntry result;
        CHECK_FOR_DEVICE_ERRORS(cudaMemcpy(&result.objectStatistics, _objectStatisticsEntry, sizeof(ObjectStatisticsEntry), cudaMemcpyDeviceToHost));

        auto control = getLineageMapControl();
        result.lineageEntries.resize(control.numCompactEntries);
        if (control.numCompactEntries > 0) {
            CHECK_FOR_DEVICE_ERRORS(
                cudaMemcpy(result.lineageEntries.data(), _lineageStatisticsEntries, sizeof(LineageStatisticsEntry) * control.numCompactEntries, cudaMemcpyDeviceToHost));
        }
        std::sort(result.lineageEntries.begin(), result.lineageEntries.end(), [](auto const& lhs, auto const& rhs) { return lhs.lineageId < rhs.lineageId; });
        return result;
    }

    __host__ bool isLineageAccumulatorGCNeeded() const
    {
        auto control = getLineageMapControl();
        return control.numUsedAccumulatorSlots[control.activeAccumulatorBuffer] > static_cast<uint32_t>(_lineageArrayCapacity) / 2;
    }

    //object statistics (timestep)
    __inline__ __device__ void resetObjectStatistics()
    {
        auto& objectStatistics = *_objectStatisticsEntry;
        objectStatistics.numSolidObjects = 0;
        objectStatistics.numFluidObjects = 0;
        objectStatistics.numFreeCellObjects = 0;
        objectStatistics.numCellObjects = 0;
        objectStatistics.numEnergyParticles = 0;
        objectStatistics.totalInternalEnergy = 0;
    }
    __inline__ __device__ void incNumSolidObjects() { atomicAdd(&_objectStatisticsEntry->numSolidObjects, 1u); }
    __inline__ __device__ void incNumFluidObjects() { atomicAdd(&_objectStatisticsEntry->numFluidObjects, 1u); }
    __inline__ __device__ void incNumFreeCellObjects() { atomicAdd(&_objectStatisticsEntry->numFreeCellObjects, 1u); }
    __inline__ __device__ void incNumCellObjects() { atomicAdd(&_objectStatisticsEntry->numCellObjects, 1u); }
    __inline__ __device__ void incNumEnergyParticles() { atomicAdd(&_objectStatisticsEntry->numEnergyParticles, 1u); }
    __inline__ __device__ void addInternalEnergy(float value) { atomicAdd(&_objectStatisticsEntry->totalInternalEnergy, toDouble(value)); }

    //lineage statistics
    __inline__ __device__ int getLineageMapCapacity() const { return _lineageArrayCapacity; }
    __inline__ __device__ void resetLineageMapSlot(int index)
    {
        auto& slot = _lineageMap[index];
        slot.lineageId = LineageIdEmpty;
        slot.colorBitset = 0;
        slot.numCreatures = 0;
        slot.numGenomes = 0;
        slot.sumCreatureCells = 0;
        slot.sumCreatureGenerations = 0;
        slot.sumGenomeNodes = 0;
        slot.sumMutationRates = 0;
        slot.sumCreatureEnergy = 0;
        slot.maxCreatureGeneration = 0;
        slot.representativeCellId = 0;
    }
    __inline__ __device__ void resetCompactLineageCounter() { _lineageAccumulatorMapControl->numCompactEntries = 0; }

    __inline__ __device__ int insertOrFindLineageSlot(uint32_t lineageId)
    {
        auto mask = _lineageArrayCapacity - 1;
        auto index = toInt((lineageId * 2654435761u) & mask);
        for (int i = 0; i < _lineageArrayCapacity; ++i) {
            auto origLineageId = atomicCAS(&_lineageMap[index].lineageId, LineageIdEmpty, lineageId);
            if (origLineageId == LineageIdEmpty || origLineageId == lineageId) {
                return index;
            }
            index = (index + 1) & mask;
        }
        return -1;
    }
    __inline__ __device__ int findLineageSlot(uint32_t lineageId) const
    {
        auto mask = _lineageArrayCapacity - 1;
        auto index = toInt((lineageId * 2654435761u) & mask);
        for (int i = 0; i < _lineageArrayCapacity; ++i) {
            auto slotLineageId = _lineageMap[index].lineageId;
            if (slotLineageId == lineageId) {
                return index;
            }
            if (slotLineageId == LineageIdEmpty) {
                return -1;
            }
            index = (index + 1) & mask;
        }
        return -1;
    }
    __inline__ __device__ void addLineageCreatureData(int slotIndex, uint32_t numCells, uint32_t generation)
    {
        auto& slot = _lineageMap[slotIndex];
        atomicAdd(&slot.numCreatures, 1u);
        atomicAdd(&slot.sumCreatureCells, toFloat(numCells));
        atomicAdd(&slot.sumCreatureGenerations, toFloat(generation));
        atomicMax(&slot.maxCreatureGeneration, generation);
    }
    __inline__ __device__ void updateLineageRepresentativeCell(int slotIndex, uint32_t generation, uint64_t cellId)
    {
        auto& slot = _lineageMap[slotIndex];
        if (generation == slot.maxCreatureGeneration) {
            atomicCAS(reinterpret_cast<unsigned long long*>(&slot.representativeCellId), 0ull, static_cast<unsigned long long>(cellId));
        }
    }
    __inline__ __device__ void addLineageGenomeData(int slotIndex, float numNodes, float meanMutationRate, uint32_t nodeColorBitset)
    {
        auto& slot = _lineageMap[slotIndex];
        atomicAdd(&slot.numGenomes, 1u);
        atomicAdd(&slot.sumGenomeNodes, numNodes);
        atomicAdd(&slot.sumMutationRates, meanMutationRate);
        atomicOr(&slot.colorBitset, nodeColorBitset);
    }
    __inline__ __device__ void addLineageEnergy(int slotIndex, float energy)
    {
        auto& slot = _lineageMap[slotIndex];
        atomicAdd(&slot.sumCreatureEnergy, energy);
    }
    __inline__ __device__ void compactLineageSlot(int index)
    {
        auto const& slot = _lineageMap[index];
        if (slot.lineageId == LineageIdEmpty || slot.numCreatures == 0) {
            return;
        }
        auto entryIndex = atomicAdd(&_lineageAccumulatorMapControl->numCompactEntries, 1u);
        auto& entry = _lineageStatisticsEntries[entryIndex];
        entry.lineageId = slot.lineageId;
        entry.colorBitset = slot.colorBitset;
        entry.numCreatures = slot.numCreatures;
        entry.numGenomes = slot.numGenomes;
        entry.sumCreatureCells = slot.sumCreatureCells;
        entry.sumCreatureGenerations = slot.sumCreatureGenerations;
        entry.sumGenomeNodes = slot.sumGenomeNodes;
        entry.sumMutationRates = slot.sumMutationRates;
        entry.sumCreatureEnergy = slot.sumCreatureEnergy;
        entry.representativeCellId = slot.representativeCellId;
        entry.numCreatedCreatures = 0;
        entry.totalMutations = 0;
        auto accumulatorIndex = findAccumulatorSlot(slot.lineageId);
        if (accumulatorIndex != -1) {
            auto const& accumulatorSlot = getActiveAccumulatorMap()[accumulatorIndex];
            entry.numCreatedCreatures = accumulatorSlot.numCreatedCreatures;
            entry.totalMutations = accumulatorSlot.totalMutations;
        }
    }

    //lineage accumulator map (persistent, garbage-collected occasionally)
    __inline__ __device__ void resetInactiveAccumulatorSlot(int index)
    {
        auto& slot = _lineageAccumulatorMaps[1 - _lineageAccumulatorMapControl->activeAccumulatorBuffer][index];
        slot.lineageId = LineageIdEmpty;
        slot.numCreatedCreatures = 0;
        slot.totalMutations = 0;
        if (index == 0) {
            _lineageAccumulatorMapControl->numUsedAccumulatorSlots[1 - _lineageAccumulatorMapControl->activeAccumulatorBuffer] = 0;
        }
    }
    __inline__ __device__ void migrateActiveAccumulatorSlot(int index)
    {
        auto activeBuffer = _lineageAccumulatorMapControl->activeAccumulatorBuffer;
        auto const& slot = _lineageAccumulatorMaps[activeBuffer][index];
        if (slot.lineageId == LineageIdEmpty) {
            return;
        }
        if (findLineageSlot(slot.lineageId) == -1) {
            return;
        }
        auto targetIndex = insertAccumulatorSlot(1 - activeBuffer, slot.lineageId);
        if (targetIndex != -1) {
            auto& targetSlot = _lineageAccumulatorMaps[1 - activeBuffer][targetIndex];
            targetSlot.numCreatedCreatures = slot.numCreatedCreatures;
            targetSlot.totalMutations = slot.totalMutations;
        }
    }
    __inline__ __device__ void flipAccumulatorBuffers() { _lineageAccumulatorMapControl->activeAccumulatorBuffer = 1 - _lineageAccumulatorMapControl->activeAccumulatorBuffer; }

    __inline__ __device__ void incCreatedCreature(uint32_t lineageId)
    {
        auto slotIndex = findOrInsertAccumulatorSlot(lineageId);
        if (slotIndex != -1) {
            atomicAdd(&getActiveAccumulatorMap()[slotIndex].numCreatedCreatures, 1ull);
        }
    }
    __inline__ __device__ void addMutations(uint32_t lineageId, float value)
    {
        auto slotIndex = findOrInsertAccumulatorSlot(lineageId);
        if (slotIndex != -1) {
            atomicAdd(&getActiveAccumulatorMap()[slotIndex].totalMutations, value);
        }
    }

private:
    static auto constexpr LineageIdEmpty = 0xffffffffu;

    struct LineageMapEntry
    {
        uint32_t lineageId;  // LineageIdEmpty = slot is unused
        uint32_t colorBitset;
        uint32_t numCreatures;
        uint32_t numGenomes;
        float sumCreatureCells;
        float sumCreatureGenerations;
        float sumGenomeNodes;
        float sumMutationRates;
        float sumCreatureEnergy;
        uint32_t maxCreatureGeneration;
        uint64_t representativeCellId;  // Cell of a creature with the highest generation; 0 = not set
    };
    struct LineageAccumulatorMapEntry
    {
        uint32_t lineageId;  // LineageIdEmpty = slot is unused
        unsigned long long numCreatedCreatures;
        double totalMutations;
    };
    struct LineageAccumulatorMapControl
    {
        uint32_t numCompactEntries;
        uint32_t activeAccumulatorBuffer;
        uint32_t numUsedAccumulatorSlots[2];
    };

    __host__ LineageAccumulatorMapControl getLineageMapControl() const
    {
        LineageAccumulatorMapControl result;
        CHECK_FOR_DEVICE_ERRORS(cudaMemcpy(&result, _lineageAccumulatorMapControl, sizeof(LineageAccumulatorMapControl), cudaMemcpyDeviceToHost));
        return result;
    }

    __inline__ __device__ LineageAccumulatorMapEntry* getActiveAccumulatorMap() { return _lineageAccumulatorMaps[_lineageAccumulatorMapControl->activeAccumulatorBuffer]; }

    __inline__ __device__ int insertAccumulatorSlot(uint32_t bufferIndex, uint32_t lineageId)
    {
        auto map = _lineageAccumulatorMaps[bufferIndex];
        auto mask = _lineageArrayCapacity - 1;
        auto index = toInt((lineageId * 2654435761u) & mask);
        for (int i = 0; i < _lineageArrayCapacity; ++i) {
            auto origLineageId = atomicCAS(&map[index].lineageId, LineageIdEmpty, lineageId);
            if (origLineageId == LineageIdEmpty) {
                atomicAdd(&_lineageAccumulatorMapControl->numUsedAccumulatorSlots[bufferIndex], 1u);
                return index;
            }
            if (origLineageId == lineageId) {
                return index;
            }
            index = (index + 1) & mask;
        }
        return -1;
    }
    __inline__ __device__ int findOrInsertAccumulatorSlot(uint32_t lineageId)
    {
        return insertAccumulatorSlot(_lineageAccumulatorMapControl->activeAccumulatorBuffer, lineageId);
    }
    __inline__ __device__ int findAccumulatorSlot(uint32_t lineageId)
    {
        auto map = getActiveAccumulatorMap();
        auto mask = _lineageArrayCapacity - 1;
        auto index = toInt((lineageId * 2654435761u) & mask);
        for (int i = 0; i < _lineageArrayCapacity; ++i) {
            auto slotLineageId = map[index].lineageId;
            if (slotLineageId == lineageId) {
                return index;
            }
            if (slotLineageId == LineageIdEmpty) {
                return -1;
            }
            index = (index + 1) & mask;
        }
        return -1;
    }

    int _lineageArrayCapacity;  // Used for all lineage maps and arrays

    ObjectStatisticsEntry* _objectStatisticsEntry;
    LineageStatisticsEntry* _lineageStatisticsEntries;

    // Lineage map for timestep values
    LineageMapEntry* _lineageMap;

    // Lineage map for accumulated values (with history => needs migration)
    LineageAccumulatorMapControl* _lineageAccumulatorMapControl;
    LineageAccumulatorMapEntry* _lineageAccumulatorMaps[2];
};
