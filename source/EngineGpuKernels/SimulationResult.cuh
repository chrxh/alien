#pragma once

class SimulationResult
{
public:
    __host__ void init()
    {
        CudaMemoryManager::getInstance().acquireMemory<bool>(1, _arrayResizingNeeded);
        CudaMemoryManager::getInstance().acquireMemory<ProcessMonitorData>(1, _statistics);
        ProcessMonitorData statistics;
        CHECK_FOR_CUDA_ERROR(cudaMemcpy(_statistics, &statistics, sizeof(ProcessMonitorData), cudaMemcpyHostToDevice));
    }

    __host__ void free() {
        CudaMemoryManager::getInstance().freeMemory(_statistics);
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
    __host__ ProcessMonitorData getProcessMonitorData()
    {
        ProcessMonitorData result;
        CHECK_FOR_CUDA_ERROR(cudaMemcpy(&result, _statistics, sizeof(ProcessMonitorData), cudaMemcpyDeviceToHost));
        ProcessMonitorData empty;
        CHECK_FOR_CUDA_ERROR(cudaMemcpy(_statistics, &empty, sizeof(ProcessMonitorData), cudaMemcpyHostToDevice));
        return result;
    }

    __device__ void setArrayResizeNeeded(bool value) { *_arrayResizingNeeded = value; }

    __device__ void resetStatistics() { *_statistics = ProcessMonitorData(); }
    __device__ void incCreatedCell() { atomicAdd(&_statistics->createdCells, 1); }
    __device__ void incSuccessfulAttack() { atomicAdd(&_statistics->sucessfulAttacks, 1); }
    __device__ void incFailedAttack() { atomicAdd(&_statistics->failedAttacks, 1); }
    __device__ void incMuscleActivity() { atomicAdd(&_statistics->muscleActivities, 1); }

private:
    ProcessMonitorData* _statistics;
    bool* _arrayResizingNeeded;
};