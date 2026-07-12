#pragma once

#include <EngineInterface/LineageStatistics.h>
#include <EngineInterface/StatisticsRawData.h>

#include "Base.cuh"
#include "CudaMemoryManager.cuh"

class SimulationStatistics
{
public:
    static auto constexpr DefaultLineageMapCapacity = 1 << 18;  // Power of two
    static auto constexpr LineageIdEmpty = 0xffffffffu;

    __host__ void init(int lineageMapCapacity = DefaultLineageMapCapacity)
    {
        _lineageMapCapacity = lineageMapCapacity;
        CudaMemoryManager::getInstance().acquireMemory<StatisticsRawData>(1, _data);
        CudaMemoryManager::getInstance().acquireMemory<MutantStatistics>(MutantToColorCountMapSize, _mutantToMutantStatisticsMap);
        CudaMemoryManager::getInstance().acquireMemory<LineageMapSlot>(_lineageMapCapacity, _lineageMap);
        CudaMemoryManager::getInstance().acquireMemory<LineageStatisticsEntry>(_lineageMapCapacity, _lineageCompactData);
        CudaMemoryManager::getInstance().acquireMemory<LineageAccumulatorSlot>(_lineageMapCapacity, _lineageAccumulatorMaps[0]);
        CudaMemoryManager::getInstance().acquireMemory<LineageAccumulatorSlot>(_lineageMapCapacity, _lineageAccumulatorMaps[1]);
        CudaMemoryManager::getInstance().acquireMemory<LineageMapControl>(1, _lineageMapControl);
        CHECK_FOR_DEVICE_ERRORS(cudaMemset(_data, 0, sizeof(StatisticsRawData)));
        CHECK_FOR_DEVICE_ERRORS(cudaMemset(_lineageMapControl, 0, sizeof(LineageMapControl)));

        // Values must start at zero; the lineageId key column (first member) is set to LineageIdEmpty
        CHECK_FOR_DEVICE_ERRORS(cudaMemset(_lineageMap, 0, sizeof(LineageMapSlot) * _lineageMapCapacity));
        CHECK_FOR_DEVICE_ERRORS(cudaMemset2D(_lineageMap, sizeof(LineageMapSlot), 0xff, sizeof(uint32_t), _lineageMapCapacity));
        for (int i = 0; i < 2; ++i) {
            CHECK_FOR_DEVICE_ERRORS(cudaMemset(_lineageAccumulatorMaps[i], 0, sizeof(LineageAccumulatorSlot) * _lineageMapCapacity));
            CHECK_FOR_DEVICE_ERRORS(cudaMemset2D(_lineageAccumulatorMaps[i], sizeof(LineageAccumulatorSlot), 0xff, sizeof(uint32_t), _lineageMapCapacity));
        }
    }

    __host__ void free()
    {
        CudaMemoryManager::getInstance().freeMemory(_data);
        CudaMemoryManager::getInstance().freeMemory(_mutantToMutantStatisticsMap);
        CudaMemoryManager::getInstance().freeMemory(_lineageMap);
        CudaMemoryManager::getInstance().freeMemory(_lineageCompactData);
        CudaMemoryManager::getInstance().freeMemory(_lineageAccumulatorMaps[0]);
        CudaMemoryManager::getInstance().freeMemory(_lineageAccumulatorMaps[1]);
        CudaMemoryManager::getInstance().freeMemory(_lineageMapControl);
    }

    __host__ StatisticsRawData getStatistics()
    {
        StatisticsRawData result;
        CHECK_FOR_DEVICE_ERRORS(cudaMemcpy(&result, _data, sizeof(StatisticsRawData), cudaMemcpyDeviceToHost));
        return result;
    }

    __host__ RawLineageStatistics getLineageStatistics()
    {
        auto control = getLineageMapControl();
        RawLineageStatistics result;
        result.entries.resize(control.numCompactEntries);
        if (control.numCompactEntries > 0) {
            CHECK_FOR_DEVICE_ERRORS(
                cudaMemcpy(result.entries.data(), _lineageCompactData, sizeof(LineageStatisticsEntry) * control.numCompactEntries, cudaMemcpyDeviceToHost));
        }
        return result;
    }

    __host__ bool isLineageAccumulatorGCNeeded() const
    {
        auto control = getLineageMapControl();
        return control.numUsedAccumulatorSlots[control.activeAccumulatorBuffer] > static_cast<uint32_t>(_lineageMapCapacity) / 2;
    }

    //timestep statistics
    __inline__ __device__ void resetTimestepData()
    {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            // Evolution statistics are collected at deterministic timesteps (separate kernel sequence)
            // and must survive the wall-clock-throttled timestep statistics updates.
            auto evolution = _data->timeline.timestep.evolution;
            _data->timeline.timestep = TimestepStatistics();
            _data->timeline.timestep.evolution = evolution;
        }
        auto partition = calcSystemThreadPartition(MutantToColorCountMapSize);
        for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
            _mutantToMutantStatisticsMap[index].count = 0;
            _mutantToMutantStatisticsMap[index].numObjects = 0;
            _mutantToMutantStatisticsMap[index].color = 0;
        }
    }

    __inline__ __device__ void incNumObjects(int color) { atomicAdd(&(_data->timeline.timestep.numObjects[color]), 1); }
    __inline__ __device__ void incNumReplicator(int color) { atomicAdd(&_data->timeline.timestep.numSelfReplicators[color], 1); }
    __inline__ __device__ int getNumReplicators()
    {
        auto result = 0;
        for (int i = 0; i < MAX_COLORS; ++i) {
            result += _data->timeline.timestep.numSelfReplicators[i];
        }
        return result;
    }
    __inline__ __device__ void incNumViruses(int color) { atomicAdd(&_data->timeline.timestep.numViruses[color], 1); }
    __inline__ __device__ void incNumFreeCells(int color) { atomicAdd(&_data->timeline.timestep.numFreeCells[color], 1); }
    __inline__ __device__ void incNumParticles(int color) { atomicAdd(&_data->timeline.timestep.numEnergyParticles[color], 1); }
    __inline__ __device__ void addEnergy(int color, float valueToAdd) { atomicAdd(&_data->timeline.timestep.totalEnergy[color], valueToAdd); }
    __inline__ __device__ void addNumObjects(int color, float valueToAdd) { atomicAdd(&_data->timeline.timestep.numObjects[color], valueToAdd); }
    __inline__ __device__ double getSummedNumCells()
    {
        auto result = 0.0;
        for (int i = 0; i < MAX_COLORS; ++i) {
            result += toDouble(_data->timeline.timestep.numObjects[i]);
        }
        return result;
    }
    __inline__ __device__ void incMutant(int color, uint32_t lineageId, float numCells)
    {
        atomicAdd(&_mutantToMutantStatisticsMap[lineageId % MutantToColorCountMapSize].count, 1);
        atomicMax(&_mutantToMutantStatisticsMap[lineageId % MutantToColorCountMapSize].color, color);
        atomicAdd(&_mutantToMutantStatisticsMap[lineageId % MutantToColorCountMapSize].numObjects, numCells);
    }
    __inline__ __device__ void halveNumConnections()
    {
        for (int i = 0; i < MAX_COLORS; ++i) {
            _data->timeline.timestep.numFreeCells[i] /= 2;
        }
    }

    //evolution statistics (timestep)
    __inline__ __device__ void resetEvolutionStatistics() { _data->timeline.timestep.evolution = EvolutionStatistics(); }
    __inline__ __device__ void incNumSolidObjects() { atomicAdd(&_data->timeline.timestep.evolution.numSolidObjects, 1); }
    __inline__ __device__ void incNumFluidObjects() { atomicAdd(&_data->timeline.timestep.evolution.numFluidObjects, 1); }
    __inline__ __device__ void incNumCellObjects() { atomicAdd(&_data->timeline.timestep.evolution.numCellObjects, 1); }
    __inline__ __device__ void addCreatureStatistics(uint32_t numCells, uint32_t generation, float accumulatedMutations)
    {
        auto& evolution = _data->timeline.timestep.evolution;
        atomicAdd(&evolution.numCreatures, 1);
        atomicAdd(&evolution.sumCreatureCells, toFloat(numCells));
        atomicAdd(&evolution.sumCreatureGenerations, toFloat(generation));
        atomicAdd(&evolution.sumAccumulatedMutations, accumulatedMutations);
    }
    __inline__ __device__ void addGenomeStatistics(float numNodes, float meanMutationRate)
    {
        auto& evolution = _data->timeline.timestep.evolution;
        atomicAdd(&evolution.numGenomes, 1);
        atomicAdd(&evolution.sumGenomeNodes, numNodes);
        atomicAdd(&evolution.sumMutationRates, meanMutationRate);
    }
    __inline__ __device__ void addCreatureEnergy(float value) { atomicAdd(&_data->timeline.timestep.evolution.sumCreatureEnergy, value); }

    //lineage statistics
    __inline__ __device__ int getLineageMapCapacity() const { return _lineageMapCapacity; }
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
    __inline__ __device__ void resetCompactLineageCounter() { _lineageMapControl->numCompactEntries = 0; }

    __inline__ __device__ int insertOrFindLineageSlot(uint32_t lineageId)
    {
        auto mask = _lineageMapCapacity - 1;
        auto index = toInt((lineageId * 2654435761u) & mask);
        for (int i = 0; i < _lineageMapCapacity; ++i) {
            auto origLineageId = atomicCAS(&_lineageMap[index].lineageId, LineageIdEmpty, lineageId);
            if (origLineageId == LineageIdEmpty || origLineageId == lineageId) {
                return index;
            }
            index = (index + 1) & mask;
        }
        _data->timeline.timestep.evolution.lineageMapOverflow = 1;
        return -1;
    }
    __inline__ __device__ int findLineageSlot(uint32_t lineageId) const
    {
        auto mask = _lineageMapCapacity - 1;
        auto index = toInt((lineageId * 2654435761u) & mask);
        for (int i = 0; i < _lineageMapCapacity; ++i) {
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
        auto entryIndex = atomicAdd(&_lineageMapControl->numCompactEntries, 1u);
        auto& entry = _lineageCompactData[entryIndex];
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
    __inline__ __device__ void finalizeLineageStatistics()
    {
        _data->timeline.timestep.evolution.numActiveLineages = toInt(_lineageMapControl->numCompactEntries);
    }

    //lineage accumulator map (persistent, garbage-collected occasionally)
    __inline__ __device__ void resetInactiveAccumulatorSlot(int index)
    {
        auto& slot = _lineageAccumulatorMaps[1 - _lineageMapControl->activeAccumulatorBuffer][index];
        slot.lineageId = LineageIdEmpty;
        slot.numCreatedCreatures = 0;
        slot.totalMutations = 0;
        if (index == 0) {
            _lineageMapControl->numUsedAccumulatorSlots[1 - _lineageMapControl->activeAccumulatorBuffer] = 0;
        }
    }
    __inline__ __device__ void migrateActiveAccumulatorSlot(int index)
    {
        auto activeBuffer = _lineageMapControl->activeAccumulatorBuffer;
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
    __inline__ __device__ void flipAccumulatorBuffers() { _lineageMapControl->activeAccumulatorBuffer = 1 - _lineageMapControl->activeAccumulatorBuffer; }

    //accumulated statistics
    __host__ void resetAccumulatedStatistics()
    {
        StatisticsRawData hostData;
        CHECK_FOR_DEVICE_ERRORS(cudaMemcpy(&hostData, _data, sizeof(StatisticsRawData), cudaMemcpyDeviceToHost));
        hostData.timeline.accumulated = AccumulatedStatistics();
        CHECK_FOR_DEVICE_ERRORS(cudaMemcpy(_data, &hostData, sizeof(StatisticsRawData), cudaMemcpyHostToDevice));
    }

    __inline__ __device__ void incNumCreatedCells(int color) { alienAtomicAdd64(&_data->timeline.accumulated.numCreatedCells[color], uint64_t(1)); }
    __inline__ __device__ void incNumCreatedReplicators(int color) { alienAtomicAdd64(&_data->timeline.accumulated.numCreatedReplicators[color], uint64_t(1)); }
    __inline__ __device__ void incNumAttacks(int color) { alienAtomicAdd64(&_data->timeline.accumulated.numAttacks[color], uint64_t(1)); }
    __inline__ __device__ void incNumMuscleActivities(int color) { alienAtomicAdd64(&_data->timeline.accumulated.numMuscleActivities[color], uint64_t(1)); }
    __inline__ __device__ void incNumDefenderActivities(int color) { alienAtomicAdd64(&_data->timeline.accumulated.numDefenderActivities[color], uint64_t(1)); }
    __inline__ __device__ void incNumDepotActivities(int color) { alienAtomicAdd64(&_data->timeline.accumulated.numDepotActivities[color], uint64_t(1)); }
    __inline__ __device__ void incNumInjectionActivities(int color)
    {
        alienAtomicAdd64(&_data->timeline.accumulated.numInjectionActivities[color], uint64_t(1));
    }
    __inline__ __device__ void incNumCompletedInjections(int color)
    {
        alienAtomicAdd64(&_data->timeline.accumulated.numCompletedInjections[color], uint64_t(1));
    }
    __inline__ __device__ void incNumGeneratorPulses(int color) { alienAtomicAdd64(&_data->timeline.accumulated.numGeneratorPulses[color], uint64_t(1)); }
    __inline__ __device__ void incNumNeuronActivities(int color) { alienAtomicAdd64(&_data->timeline.accumulated.numNeuronActivities[color], uint64_t(1)); }
    __inline__ __device__ void incNumSensorActivities(int color) { alienAtomicAdd64(&_data->timeline.accumulated.numSensorActivities[color], uint64_t(1)); }
    __inline__ __device__ void incNumSensorMatches(int color) { alienAtomicAdd64(&_data->timeline.accumulated.numSensorMatches[color], uint64_t(1)); }
    __inline__ __device__ void incNumReconnectorCreated(int color) { alienAtomicAdd64(&_data->timeline.accumulated.numReconnectorCreated[color], uint64_t(1)); }
    __inline__ __device__ void incNumReconnectorRemoved(int color) { alienAtomicAdd64(&_data->timeline.accumulated.numReconnectorRemoved[color], uint64_t(1)); }
    __inline__ __device__ void incNumDetonations(int color) { alienAtomicAdd64(&_data->timeline.accumulated.numDetonations[color], uint64_t(1)); }

    __inline__ __device__ void incCreatedCreature(uint32_t lineageId)
    {
        alienAtomicAdd64(&_data->timeline.accumulated.numCreatedCreatures, uint64_t(1));
        auto slotIndex = findOrInsertAccumulatorSlot(lineageId);
        if (slotIndex >= 0) {
            atomicAdd(&getActiveAccumulatorMap()[slotIndex].numCreatedCreatures, 1u);
        }
    }
    __inline__ __device__ void addMutations(uint32_t lineageId, float value)
    {
        atomicAdd(&_data->timeline.accumulated.totalMutations, toDouble(value));
        auto slotIndex = findOrInsertAccumulatorSlot(lineageId);
        if (slotIndex >= 0) {
            atomicAdd(&getActiveAccumulatorMap()[slotIndex].totalMutations, value);
        }
    }

    //histogram
    __inline__ __device__ void resetHistogramData()
    {
        _data->histogram.maxAge = 0;
        for (int i = 0; i < MAX_COLORS; ++i) {
            for (int j = 0; j < MAX_HISTOGRAM_SLOTS; ++j) {
                _data->histogram.numCellsByColorBySlot[i][j] = 0;
            }
        }
    }
    __inline__ __device__ void incNumCells(int color, int slot) { atomicAdd(&(_data->histogram.numCellsByColorBySlot[color][slot]), 1); }
    __inline__ __device__ void maxAge(int value) { atomicMax(&_data->histogram.maxAge, value); }
    __inline__ __device__ int getMaxAge() const { return _data->histogram.maxAge; }

private:
    struct LineageMapSlot
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
    struct LineageAccumulatorSlot
    {
        uint32_t lineageId;  // LineageIdEmpty = slot is unused
        uint32_t numCreatedCreatures;
        float totalMutations;
    };
    struct LineageMapControl
    {
        uint32_t numCompactEntries;
        uint32_t activeAccumulatorBuffer;
        uint32_t numUsedAccumulatorSlots[2];
    };

    __host__ LineageMapControl getLineageMapControl() const
    {
        LineageMapControl result;
        CHECK_FOR_DEVICE_ERRORS(cudaMemcpy(&result, _lineageMapControl, sizeof(LineageMapControl), cudaMemcpyDeviceToHost));
        return result;
    }

    __inline__ __device__ LineageAccumulatorSlot* getActiveAccumulatorMap() { return _lineageAccumulatorMaps[_lineageMapControl->activeAccumulatorBuffer]; }

    __inline__ __device__ int insertAccumulatorSlot(uint32_t bufferIndex, uint32_t lineageId)
    {
        auto map = _lineageAccumulatorMaps[bufferIndex];
        auto mask = _lineageMapCapacity - 1;
        auto index = toInt((lineageId * 2654435761u) & mask);
        for (int i = 0; i < _lineageMapCapacity; ++i) {
            auto origLineageId = atomicCAS(&map[index].lineageId, LineageIdEmpty, lineageId);
            if (origLineageId == LineageIdEmpty) {
                atomicAdd(&_lineageMapControl->numUsedAccumulatorSlots[bufferIndex], 1u);
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
        return insertAccumulatorSlot(_lineageMapControl->activeAccumulatorBuffer, lineageId);
    }
    __inline__ __device__ int findAccumulatorSlot(uint32_t lineageId)
    {
        auto map = getActiveAccumulatorMap();
        auto mask = _lineageMapCapacity - 1;
        auto index = toInt((lineageId * 2654435761u) & mask);
        for (int i = 0; i < _lineageMapCapacity; ++i) {
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

    StatisticsRawData* _data;
    int _lineageMapCapacity;

    LineageMapSlot* _lineageMap;
    LineageStatisticsEntry* _lineageCompactData;
    LineageAccumulatorSlot* _lineageAccumulatorMaps[2];
    LineageMapControl* _lineageMapControl;

    //for diversity calculation
    static auto constexpr MutantToColorCountMapSize = 1 << 20;
    struct MutantStatistics
    {
        uint32_t color;
        uint32_t count;
        float numObjects;
    };
    MutantStatistics* _mutantToMutantStatisticsMap;
};
