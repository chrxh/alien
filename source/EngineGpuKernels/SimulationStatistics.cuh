#pragma once

#include "EngineInterface/StatisticsData.h"

#include "Base.cuh"
#include "Definitions.cuh"

class SimulationStatistics
{
public:

    __host__ void init()
    {
        CudaMemoryManager::getInstance().acquireMemory<StatisticsData>(1, _data);
        CHECK_FOR_CUDA_ERROR(cudaMemset(_data, 0, sizeof(StatisticsData)));
    }

    __host__ void free()
    {
        CudaMemoryManager::getInstance().freeMemory(_data);
    }

    __host__ StatisticsData getStatistics()
    {
        StatisticsData result;
        CHECK_FOR_CUDA_ERROR(cudaMemcpy(&result, _data, sizeof(StatisticsData), cudaMemcpyDeviceToHost));
        return result;
    }

    //accumulated statistics
    __host__ void resetAccumulatedStatistics()
    {
        StatisticsData hostData;
        CHECK_FOR_CUDA_ERROR(cudaMemcpy(&hostData, _data, sizeof(StatisticsData), cudaMemcpyDeviceToHost));
        hostData.timeline.accumulated = AccumulatedStatistics();
        CHECK_FOR_CUDA_ERROR(cudaMemcpy(_data, &hostData, sizeof(StatisticsData), cudaMemcpyHostToDevice));
    }

    __device__ void incCreatedCell() { alienAtomicAdd64(&_data->timeline.accumulated.numCreatedCells, uint64_t(1)); }
    __device__ void incSuccessfulAttack() { alienAtomicAdd64(&_data->timeline.accumulated.numSuccessfulAttacks, uint64_t(1)); }
    __device__ void incFailedAttack() { alienAtomicAdd64(&_data->timeline.accumulated.numFailedAttacks, uint64_t(1)); }
    __device__ void incMuscleActivity() { alienAtomicAdd64(&_data->timeline.accumulated.numMuscleActivities, uint64_t(1)); }


    //timestep statistics
    __inline__ __device__ void resetTimestepData()
    {
        _data->timeline.timestep = TimestepStatistics();
    }

    __inline__ __device__ void incNumCell(int color) { atomicAdd(&(_data->timeline.timestep.numCellsByColor[color]), 1); }
    __inline__ __device__ void incNumConnections(int numConnections) { atomicAdd(&_data->timeline.timestep.numConnections, numConnections); }
    __inline__ __device__ void setNumParticles(int value) { _data->timeline.timestep.numParticles = value; }
    __inline__ __device__ void halveNumConnections() { _data->timeline.timestep.numConnections /= 2; }

    //histogram
    __inline__ __device__ void resetHistogramData()
    {
        _data->histogram.maxValue = 0;
        for (int i = 0; i < MAX_COLORS; ++i) {
            for (int j = 0; j < MAX_HISTOGRAM_SLOTS; ++j) {
                _data->histogram.numCellsByColorBySlot[i][j] = 0;
            }
        }
    }
    __inline__ __device__ void incNumCell(int color, int slot) { atomicAdd(&(_data->histogram.numCellsByColorBySlot[color][slot]), 1); }
    __inline__ __device__ void maxValue(int value) { atomicMax(&_data->histogram.maxValue, value); }
    __inline__ __device__ int getMaxValue() const { return _data->histogram.maxValue; }

private:
    StatisticsData* _data;
};

