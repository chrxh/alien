#pragma once

#include "Base.cuh"
#include "CudaMemoryManager.cuh"

class ActiveRadiationSources
{
public:
    __host__ __inline__ void init()
    {
        CudaMemoryManager::getInstance().acquireMemory<int>(1, _numActiveSources);
        CudaMemoryManager::getInstance().acquireMemory<int>(MAX_RADIATION_SOURCES, _activeSources);
    }

    __host__ __inline__ void free()
    {
        CudaMemoryManager::getInstance().freeMemory(_numActiveSources);
        CudaMemoryManager::getInstance().freeMemory(_activeSources);
    }

    __device__ __inline__ int getNumActiveSources() const { return *_numActiveSources; }
    __device__ __inline__ void setNumActiveSources(int value) { *_numActiveSources = value; }

    __device__ __inline__ int getActiveSource(int i) const { return _activeSources[i]; }
    __device__ __inline__ void setActiveSource(int i, int value) { _activeSources[i] = value; }

private:
    int* _numActiveSources;
    int* _activeSources;
};
