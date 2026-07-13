#pragma once

#include <EngineInterface/LineageStatistics.h>
#include <EngineInterface/OverallStatistics.h>

#include "Base.cuh"
#include "CudaMemoryManager.cuh"

class SimulationStatistics
{
public:
    __host__ void init()
    {
        _lineageArrayCapacity = 1 << 18;

        CudaMemoryManager::getInstance().acquireMemory<OverallStatisticsEntry>(1, _overallStatisticsEntry);
        CudaMemoryManager::getInstance().acquireMemory<LineageStatisticsEntry>(_lineageArrayCapacity, _lineageStatisticsEntries);

        CudaMemoryManager::getInstance().acquireMemory<LineageMapEntry>(_lineageArrayCapacity, _lineageMap);

        CudaMemoryManager::getInstance().acquireMemory<LineageAccumulatorMapControl>(1, _lineageAccumulatorMapControl);
        CudaMemoryManager::getInstance().acquireMemory<LineageAccumulatorMapEntry>(_lineageArrayCapacity, _lineageAccumulatorMaps[0]);
        CudaMemoryManager::getInstance().acquireMemory<LineageAccumulatorMapEntry>(_lineageArrayCapacity, _lineageAccumulatorMaps[1]);
        CHECK_FOR_DEVICE_ERRORS(cudaMemset(_overallStatisticsEntry, 0, sizeof(OverallStatisticsEntry)));
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
        CudaMemoryManager::getInstance().freeMemory(_overallStatisticsEntry);
        CudaMemoryManager::getInstance().freeMemory(_lineageStatisticsEntries);

        CudaMemoryManager::getInstance().freeMemory(_lineageMap);

        CudaMemoryManager::getInstance().freeMemory(_lineageAccumulatorMapControl);
        CudaMemoryManager::getInstance().freeMemory(_lineageAccumulatorMaps[0]);
        CudaMemoryManager::getInstance().freeMemory(_lineageAccumulatorMaps[1]);
    }

    __host__ OverallStatisticsEntry getOverallStatistics()
    {
        OverallStatisticsEntry result;
        CHECK_FOR_DEVICE_ERRORS(cudaMemcpy(&result, _overallStatisticsEntry, sizeof(OverallStatisticsEntry), cudaMemcpyDeviceToHost));
        return result;
    }

    __host__ LineageStatistics getLineageStatistics()
    {
        auto control = getLineageMapControl();
        LineageStatistics result;
        result.entries.resize(control.numCompactEntries);
        if (control.numCompactEntries > 0) {
            CHECK_FOR_DEVICE_ERRORS(
                cudaMemcpy(result.entries.data(), _lineageStatisticsEntries, sizeof(LineageStatisticsEntry) * control.numCompactEntries, cudaMemcpyDeviceToHost));
        }
        return result;
    }

    __host__ bool isLineageAccumulatorGCNeeded() const
    {
        auto control = getLineageMapControl();
        return control.numUsedAccumulatorSlots[control.activeAccumulatorBuffer] > static_cast<uint32_t>(_lineageArrayCapacity) / 2;
    }

    //evolution statistics (timestep)
    __inline__ __device__ void resetEvolutionStatistics()
    {
        // numCreatedCreatures and totalMutations are accumulated, never reset here
        auto& overall = *_overallStatisticsEntry;
        overall.numCreatures = 0;
        overall.numGenomes = 0;
        overall.sumCreatureCells = 0;
        overall.sumCreatureGenerations = 0;
        overall.sumGenomeNodes = 0;
        overall.sumMutationRates = 0;
        overall.sumCreatureEnergy = 0;
        overall.sumAccumulatedMutations = 0;
        overall.numSolidObjects = 0;
        overall.numFluidObjects = 0;
        overall.numCellObjects = 0;
        overall.numActiveLineages = 0;
    }
    __inline__ __device__ void incNumSolidObjects() { atomicAdd(&_overallStatisticsEntry->numSolidObjects, 1u); }
    __inline__ __device__ void incNumFluidObjects() { atomicAdd(&_overallStatisticsEntry->numFluidObjects, 1u); }
    __inline__ __device__ void incNumCellObjects() { atomicAdd(&_overallStatisticsEntry->numCellObjects, 1u); }
    __inline__ __device__ void addCreatureStatistics(uint32_t numCells, uint32_t generation, float accumulatedMutations)
    {
        auto& overall = *_overallStatisticsEntry;
        atomicAdd(&overall.numCreatures, 1u);
        atomicAdd(&overall.sumCreatureCells, toFloat(numCells));
        atomicAdd(&overall.sumCreatureGenerations, toFloat(generation));
        atomicAdd(&overall.sumAccumulatedMutations, accumulatedMutations);
    }
    __inline__ __device__ void addGenomeStatistics(float numNodes, float meanMutationRate)
    {
        auto& overall = *_overallStatisticsEntry;
        atomicAdd(&overall.numGenomes, 1u);
        atomicAdd(&overall.sumGenomeNodes, numNodes);
        atomicAdd(&overall.sumMutationRates, meanMutationRate);
    }
    __inline__ __device__ void addCreatureEnergy(float value) { atomicAdd(&_overallStatisticsEntry->sumCreatureEnergy, value); }

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
        entry.numCreatedCreatures = 0;
        entry.totalMutations = 0;
        auto accumulatorIndex = findAccumulatorSlot(slot.lineageId);
        if (accumulatorIndex >= 0) {
            auto const& accumulatorSlot = getActiveAccumulatorMap()[accumulatorIndex];
            entry.numCreatedCreatures = accumulatorSlot.numCreatedCreatures;
            entry.totalMutations = accumulatorSlot.totalMutations;
        }
    }
    __inline__ __device__ void finalizeLineageStatistics() { _overallStatisticsEntry->numActiveLineages = _lineageAccumulatorMapControl->numCompactEntries; }

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
        if (findLineageSlot(slot.lineageId) < 0) {
            return;
        }
        auto targetIndex = insertAccumulatorSlot(1 - activeBuffer, slot.lineageId);
        if (targetIndex >= 0) {
            auto& targetSlot = _lineageAccumulatorMaps[1 - activeBuffer][targetIndex];
            targetSlot.numCreatedCreatures = slot.numCreatedCreatures;
            targetSlot.totalMutations = slot.totalMutations;
        }
    }
    __inline__ __device__ void flipAccumulatorBuffers() { _lineageAccumulatorMapControl->activeAccumulatorBuffer = 1 - _lineageAccumulatorMapControl->activeAccumulatorBuffer; }

    __inline__ __device__ void incCreatedCreature(uint32_t lineageId)
    {
        atomicAdd(&_overallStatisticsEntry->numCreatedCreatures, 1u);
        auto slotIndex = findOrInsertAccumulatorSlot(lineageId);
        if (slotIndex >= 0) {
            atomicAdd(&getActiveAccumulatorMap()[slotIndex].numCreatedCreatures, 1u);
        }
    }
    __inline__ __device__ void addMutations(uint32_t lineageId, float value)
    {
        atomicAdd(&_overallStatisticsEntry->totalMutations, value);
        auto slotIndex = findOrInsertAccumulatorSlot(lineageId);
        if (slotIndex >= 0) {
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
    };
    struct LineageAccumulatorMapEntry
    {
        uint32_t lineageId;  // LineageIdEmpty = slot is unused
        uint64_t numCreatedCreatures;
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

    OverallStatisticsEntry* _overallStatisticsEntry;
    LineageStatisticsEntry* _lineageStatisticsEntries;

    // Lineage map for timestep values
    LineageMapEntry* _lineageMap;

    // Lineage map for accumulated values (with history => needs migration)
    LineageAccumulatorMapControl* _lineageAccumulatorMapControl;
    LineageAccumulatorMapEntry* _lineageAccumulatorMaps[2];
};
