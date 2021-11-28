#pragma once

class SimulationResult
{
public:
    __host__ void init()
    {
        CudaMemoryManager::getInstance().acquireMemory<bool>(1, _arrayResizingNeeded);
        CudaMemoryManager::getInstance().acquireMemory<Statistics>(1, _statistics);
        Statistics statistics;
        CHECK_FOR_CUDA_ERROR(cudaMemcpy(_statistics, &statistics, sizeof(Statistics), cudaMemcpyHostToDevice));
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

    struct Statistics
    {
        int createdCells = 0;
        int sucessfulAttacks = 0;
        int failedAttacks = 0;
        int muscleActivities = 0;
    };
    __host__ Statistics getStatistics()
    {
        Statistics result;
        CHECK_FOR_CUDA_ERROR(cudaMemcpy(&result, _statistics, sizeof(Statistics), cudaMemcpyDeviceToHost));
        return result;
    }

    __device__ void setArrayResizeNeeded(bool value) { *_arrayResizingNeeded = value; }

    __device__ void resetStatistics() { *_statistics = Statistics(); }
    __device__ void incCreatedCell() { atomicAdd(&_statistics->createdCells, 1); }
    __device__ void incSuccessfulAttack() { atomicAdd(&_statistics->sucessfulAttacks, 1); }
    __device__ void incFailedAttack() { atomicAdd(&_statistics->failedAttacks, 1); }
    __device__ void incMuscleActivity() { atomicAdd(&_statistics->muscleActivities, 1); }

private:
    Statistics* _statistics;
    bool* _arrayResizingNeeded;
};