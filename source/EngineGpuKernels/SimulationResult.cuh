#pragma once

#include "EngineInterface/MonitorData.h"

class SimulationResult
{
public:
    __host__ void init()
    {
        CudaMemoryManager::getInstance().acquireMemory<TimeIntervalMonitorData>(1, _data);
        TimeIntervalMonitorData statistics;
        CHECK_FOR_CUDA_ERROR(cudaMemcpy(_data, &statistics, sizeof(TimeIntervalMonitorData), cudaMemcpyHostToDevice));
    }

    __host__ void free() {
        CudaMemoryManager::getInstance().freeMemory(_data);
    }

    __host__ TimeIntervalMonitorData getAndResetTimeIntervalMonitorData()
    {
        TimeIntervalMonitorData result;
        CHECK_FOR_CUDA_ERROR(cudaMemcpy(&result, _data, sizeof(TimeIntervalMonitorData), cudaMemcpyDeviceToHost));
        TimeIntervalMonitorData empty;
        CHECK_FOR_CUDA_ERROR(cudaMemcpy(_data, &empty, sizeof(TimeIntervalMonitorData), cudaMemcpyHostToDevice));
        return result;
    }

    __device__ void resetStatistics() { *_data = TimeIntervalMonitorData(); }
    __device__ void incCreatedCell() { atomicAdd(&_data->numCreatedCells, 1); }
    __device__ void incSuccessfulAttack() { atomicAdd(&_data->numSuccessfulAttacks, 1); }
    __device__ void incFailedAttack() { atomicAdd(&_data->numFailedAttacks, 1); }
    __device__ void incMuscleActivity() { atomicAdd(&_data->numMuscleActivities, 1); }

private:
    TimeIntervalMonitorData* _data;
};