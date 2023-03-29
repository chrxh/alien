#pragma once

#include "EngineInterface/MonitorData.h"

#include "Base.cuh"
#include "Definitions.cuh"

using NumCellsByColor = int[MAX_COLORS];
struct CudaMonitorData
{
    NumCellsByColor numCellsByColor;
    int numConnections = 0;
    int numParticles = 0;
    double internalEnergy = 0.0;
};

class CudaMonitor
{
public:

    __host__ void init()
    {
        CudaMemoryManager::getInstance().acquireMemory<CudaMonitorData>(1, _data);
        CHECK_FOR_CUDA_ERROR(cudaMemset(_data, 0, sizeof(CudaMonitorData)));
    }

    __host__ void free()
    {
        CudaMemoryManager::getInstance().freeMemory(_data);
    }

    __host__ CudaMonitorData getMonitorData(uint64_t timeStep)
    {
        CudaMonitorData result;
        CHECK_FOR_CUDA_ERROR(cudaMemcpy(&result, _data, sizeof(CudaMonitorData), cudaMemcpyDeviceToHost));
        return result;
    }

    __inline__ __device__ void reset()
    {
        for (int i = 0; i < MAX_COLORS; ++i) {
            (_data->numCellsByColor)[i] = 0;
        }
        _data->numConnections = 0;
        _data->numParticles = 0;
        _data->internalEnergy = 0.0f;
    }

    __inline__ __device__ void incNumCell(int color) { atomicAdd(&(_data->numCellsByColor)[color], 1); }
    __inline__ __device__ void incNumConnections(int numConnections) { atomicAdd(&_data->numConnections, numConnections); }
    __inline__ __device__ void setNumParticles(int value) { _data->numParticles = value; }
    __inline__ __device__ void halveNumConnections() { _data->numConnections /= 2; }


private:
    CudaMonitorData* _data;
};

