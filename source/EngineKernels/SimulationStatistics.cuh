#pragma once

#include <EngineInterface/StatisticsRawData.h>

#include "Base.cuh"
#include "CudaMemoryManager.cuh"

class SimulationStatistics
{
public:
    __host__ void init()
    {
        CudaMemoryManager::getInstance().acquireMemory<StatisticsRawData>(1, _data);
        CudaMemoryManager::getInstance().acquireMemory<MutantStatistics>(MutantToColorCountMapSize, _mutantToMutantStatisticsMap);
        CHECK_FOR_CUDA_ERROR(cudaMemset(_data, 0, sizeof(StatisticsRawData)));
    }

    __host__ void free()
    {
        CudaMemoryManager::getInstance().freeMemory(_data);
        CudaMemoryManager::getInstance().freeMemory(_mutantToMutantStatisticsMap);
    }

    __host__ StatisticsRawData getStatistics()
    {
        StatisticsRawData result;
        CHECK_FOR_CUDA_ERROR(cudaMemcpy(&result, _data, sizeof(StatisticsRawData), cudaMemcpyDeviceToHost));
        return result;
    }

    //timestep statistics
    __inline__ __device__ void resetTimestepData()
    {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            _data->timeline.timestep = TimestepStatistics();
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

    //accumulated statistics
    __host__ void resetAccumulatedStatistics()
    {
        StatisticsRawData hostData;
        CHECK_FOR_CUDA_ERROR(cudaMemcpy(&hostData, _data, sizeof(StatisticsRawData), cudaMemcpyDeviceToHost));
        hostData.timeline.accumulated = AccumulatedStatistics();
        CHECK_FOR_CUDA_ERROR(cudaMemcpy(_data, &hostData, sizeof(StatisticsRawData), cudaMemcpyHostToDevice));
    }

    __inline__ __device__ void incNumCreatedCells(int color) { alienAtomicAdd64(&_data->timeline.accumulated.numCreatedCells[color], uint64_t(1)); }
    __inline__ __device__ void incNumCreatedReplicators(int color) { alienAtomicAdd64(&_data->timeline.accumulated.numCreatedReplicators[color], uint64_t(1)); }
    __inline__ __device__ void incNumAttacks(int color) { alienAtomicAdd64(&_data->timeline.accumulated.numAttacks[color], uint64_t(1)); }
    __inline__ __device__ void incNumMuscleActivities(int color) { alienAtomicAdd64(&_data->timeline.accumulated.numMuscleActivities[color], uint64_t(1)); }
    __inline__ __device__ void incNumDefenderActivities(int color) { alienAtomicAdd64(&_data->timeline.accumulated.numDefenderActivities[color], uint64_t(1)); }
    __inline__ __device__ void incNumDepotActivities(int color)
    {
        alienAtomicAdd64(&_data->timeline.accumulated.numDepotActivities[color], uint64_t(1));
    }
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
    StatisticsRawData* _data;

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
