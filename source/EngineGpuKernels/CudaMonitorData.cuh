#pragma once

#include "EngineInterface/MonitorData.h"

#include "Base.cuh"
#include "Definitions.cuh"
#include "Entities.cuh"

class CudaMonitorData
{
public:
    __host__ void init()
    {
        CudaMemoryManager::getInstance().acquireMemory<NumCellsByColor>(1, _numCellsByColor);
        CudaMemoryManager::getInstance().acquireMemory<int>(1, _numConnections);
        CudaMemoryManager::getInstance().acquireMemory<int>(1, _numTokens);
        CudaMemoryManager::getInstance().acquireMemory<int>(1, _numParticles);
        CudaMemoryManager::getInstance().acquireMemory<double>(1, _internalEnergy);

        CHECK_FOR_CUDA_ERROR(cudaMemset(_numCellsByColor, 0, sizeof(NumCellsByColor)));
        CHECK_FOR_CUDA_ERROR(cudaMemset(_numConnections, 0, sizeof(int)));
        CHECK_FOR_CUDA_ERROR(cudaMemset(_numTokens, 0, sizeof(int)));
        CHECK_FOR_CUDA_ERROR(cudaMemset(_numParticles, 0, sizeof(int)));

        double zero = 0.0;
        CHECK_FOR_CUDA_ERROR(cudaMemcpy(_internalEnergy, &zero, sizeof(double), cudaMemcpyHostToDevice));

    }

    __host__ void free()
    {
        CudaMemoryManager::getInstance().freeMemory(_numCellsByColor);
        CudaMemoryManager::getInstance().freeMemory(_numConnections);
        CudaMemoryManager::getInstance().freeMemory(_numTokens);
        CudaMemoryManager::getInstance().freeMemory(_numParticles);
        CudaMemoryManager::getInstance().freeMemory(_internalEnergy);
    }

    using NumCellsByColor = int[7];
    struct MonitorData
    {
        uint64_t timeStep = 0;
        NumCellsByColor numCellsByColor;
        int numConnections = 0;
        int numParticles = 0;
        int numTokens = 0;
        double totalInternalEnergy = 0.0;
    };
    __host__ MonitorData getMonitorData(uint64_t timeStep)
    {
        MonitorData result;
        CHECK_FOR_CUDA_ERROR(cudaMemcpy(&result.numCellsByColor, _numCellsByColor, sizeof(NumCellsByColor), cudaMemcpyDeviceToHost));
        CHECK_FOR_CUDA_ERROR(cudaMemcpy(&result.numConnections, _numConnections, sizeof(int), cudaMemcpyDeviceToHost));
        CHECK_FOR_CUDA_ERROR(cudaMemcpy(&result.numParticles, _numParticles, sizeof(int), cudaMemcpyDeviceToHost));
        CHECK_FOR_CUDA_ERROR(cudaMemcpy(&result.numTokens, _numTokens, sizeof(int), cudaMemcpyDeviceToHost));
        CHECK_FOR_CUDA_ERROR(cudaMemcpy(&result.totalInternalEnergy, _internalEnergy, sizeof(double), cudaMemcpyDeviceToHost));
        result.timeStep = timeStep;
        return result;
    }

    __inline__ __device__ void reset()
    {
        for (int i = 0; i < 7; ++i) {
            (*_numCellsByColor)[i] = 0;
        }
        *_numConnections = 0;
        *_numTokens = 0;
        *_numParticles = 0;
        *_internalEnergy = 0.0f;
    }

    __inline__ __device__ void incNumCell(int color) { atomicAdd(&(*_numCellsByColor)[color], 1); }
    __inline__ __device__ void incNumConnections(int numConnections) { atomicAdd(_numConnections, numConnections); }
    __inline__ __device__ void setNumParticles(int value) { *_numParticles = value; }
    __inline__ __device__ void setNumTokens(int value) { *_numTokens = value; }


/*
    __inline__ __device__ void incInternalEnergy(float changeValue)
    {
        atomicAdd(_internalEnergy, static_cast<double>(changeValue));
    }
*/

private:
    NumCellsByColor* _numCellsByColor;
    int* _numConnections;
    int* _numTokens;
    int* _numParticles;
    double* _internalEnergy;
};

