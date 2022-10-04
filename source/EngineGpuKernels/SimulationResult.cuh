#pragma once

class SimulationResult
{
public:
    __host__ void init()
    {
        CudaMemoryManager::getInstance().acquireMemory<bool>(1, _arrayResizingNeeded);
        CudaMemoryManager::getInstance().acquireMemory<ProcessMonitorData>(1, _data);
        ProcessMonitorData statistics;
        CHECK_FOR_CUDA_ERROR(cudaMemcpy(_data, &statistics, sizeof(ProcessMonitorData), cudaMemcpyHostToDevice));
    }

    __host__ void free() {
        CudaMemoryManager::getInstance().freeMemory(_data);
        CudaMemoryManager::getInstance().freeMemory(_arrayResizingNeeded);
    }

    __host__ bool isArrayResizeNeeded()
    {
        bool result;
        CHECK_FOR_CUDA_ERROR(cudaMemcpy(&result, _arrayResizingNeeded, sizeof(bool), cudaMemcpyDeviceToHost));
        return result;
    }

    struct ProcessMonitorData
    {
        int createdCells = 0;
        int sucessfulAttacks = 0;
        int failedAttacks = 0;
        int muscleActivities = 0;
    };
    __host__ ProcessMonitorData getAndResetProcessMonitorData()
    {
        ProcessMonitorData result;
        CHECK_FOR_CUDA_ERROR(cudaMemcpy(&result, _data, sizeof(ProcessMonitorData), cudaMemcpyDeviceToHost));
        ProcessMonitorData empty;
        CHECK_FOR_CUDA_ERROR(cudaMemcpy(_data, &empty, sizeof(ProcessMonitorData), cudaMemcpyHostToDevice));
        return result;
    }

    __device__ void setArrayResizeNeeded(bool value) { *_arrayResizingNeeded = value; }

    __device__ void resetStatistics() { *_data = ProcessMonitorData(); }
    __device__ void incCreatedCell() { atomicAdd(&_data->createdCells, 1); }
    __device__ void incSuccessfulAttack() { atomicAdd(&_data->sucessfulAttacks, 1); }
    __device__ void incFailedAttack() { atomicAdd(&_data->failedAttacks, 1); }
    __device__ void incMuscleActivity() { atomicAdd(&_data->muscleActivities, 1); }

private:
    ProcessMonitorData* _data;
    bool* _arrayResizingNeeded;
};