#pragma once

#include "ModelBasic/MonitorData.h"

#include "Base.cuh"
#include "Definitions.cuh"
#include "Entities.cuh"

class CudaMonitorData
{
public:
    __host__ void init()
    {
        CudaMemoryManager::getInstance().acquireMemory<int>(1, _numClusters);
        CudaMemoryManager::getInstance().acquireMemory<int>(1, _numClustersWithTokens);
        CudaMemoryManager::getInstance().acquireMemory<int>(1, _numCells);
        CudaMemoryManager::getInstance().acquireMemory<int>(1, _numTokens);
        CudaMemoryManager::getInstance().acquireMemory<int>(1, _numParticles);
        CudaMemoryManager::getInstance().acquireMemory<double>(1, _rotationalKineticEnergy);
        CudaMemoryManager::getInstance().acquireMemory<double>(1, _linearKineticEnergy);
        CudaMemoryManager::getInstance().acquireMemory<double>(1, _internalEnergy);

        checkCudaErrors(cudaMemset(_numClusters, 0, sizeof(int)));
        checkCudaErrors(cudaMemset(_numClustersWithTokens, 0, sizeof(int)));
        checkCudaErrors(cudaMemset(_numCells, 0, sizeof(int)));
        checkCudaErrors(cudaMemset(_numTokens, 0, sizeof(int)));
        checkCudaErrors(cudaMemset(_numParticles, 0, sizeof(int)));

        double zero = 0.0;
        checkCudaErrors(cudaMemcpy(_rotationalKineticEnergy, &zero, sizeof(double), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(_linearKineticEnergy, &zero, sizeof(double), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(_internalEnergy, &zero, sizeof(double), cudaMemcpyHostToDevice));

    }

    __host__ void free()
    {
        CudaMemoryManager::getInstance().freeMemory(_numClusters);
        CudaMemoryManager::getInstance().freeMemory(_numClustersWithTokens);
        CudaMemoryManager::getInstance().freeMemory(_numCells);
        CudaMemoryManager::getInstance().freeMemory(_numTokens);
        CudaMemoryManager::getInstance().freeMemory(_numParticles);
        CudaMemoryManager::getInstance().freeMemory(_rotationalKineticEnergy);
        CudaMemoryManager::getInstance().freeMemory(_linearKineticEnergy);
        CudaMemoryManager::getInstance().freeMemory(_internalEnergy);
    }

    __host__ MonitorData getMonitorData()
    {
        MonitorData result;
        checkCudaErrors(cudaMemcpy(&result.numClusters, _numClusters, sizeof(int), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(&result.numClustersWithTokens, _numClustersWithTokens, sizeof(int), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(&result.numCells, _numCells, sizeof(int), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(&result.numParticles, _numParticles, sizeof(int), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(&result.numTokens, _numTokens, sizeof(int), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(&result.totalRotationalKineticEnergy, _rotationalKineticEnergy, sizeof(double), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(&result.totalLinearKineticEnergy, _linearKineticEnergy, sizeof(double), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(&result.totalInternalEnergy, _internalEnergy, sizeof(double), cudaMemcpyDeviceToHost));
        return result;
    }

    __inline__ __device__ void reset()
    {
        *_numClusters = 0;
        *_numClustersWithTokens = 0;
        *_numCells = 0;
        *_numTokens = 0;
        *_numParticles = 0;
        *_rotationalKineticEnergy = 0.0f;
        *_linearKineticEnergy = 0.0f;
        *_internalEnergy = 0.0f;
    }

    __inline__ __device__ void incNumClusters(int changeValue)
    {
        atomicAdd(_numClusters, changeValue);
    }

    __inline__ __device__ void incNumClustersWithTokens(int changeValue)
    {
        atomicAdd(_numClustersWithTokens, changeValue);
    }

    __inline__ __device__ void incNumCells(int changeValue)
    {
        atomicAdd(_numCells, changeValue);
    }

    __inline__ __device__ void incNumParticles(int changeValue)
    {
        atomicAdd(_numParticles, changeValue);
    }

    __inline__ __device__ void incNumTokens(int changeValue)
    {
        atomicAdd(_numTokens, changeValue);
    }

    __inline__ __device__ void incRotationalKineticEnergy(float changeValue)
    {
        atomicAdd(_rotationalKineticEnergy, static_cast<double>(changeValue));
    }

    __inline__ __device__ void incLinearKineticEnergy(float changeValue)
    {
        atomicAdd(_linearKineticEnergy, static_cast<double>(changeValue));
    }

    __inline__ __device__ void incInternalEnergy(float changeValue)
    {
        atomicAdd(_internalEnergy, static_cast<double>(changeValue));
    }

private:
    int* _numClusters;
    int* _numClustersWithTokens;
    int* _numCells;
    int* _numTokens;
    int* _numParticles;
    double* _rotationalKineticEnergy;
    double* _linearKineticEnergy;
    double* _internalEnergy;
};

