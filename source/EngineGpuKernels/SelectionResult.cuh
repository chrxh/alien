#pragma once

#include "Definitions.cuh"

class SelectionResult
{
public:
    __host__ void init()
    {
        CudaMemoryManager::getInstance().acquireMemory<SelectedEntitites>(1, _selection);
        checkCudaErrors(cudaMemset(_selection, 0, sizeof(SelectedEntitites)));
    }

    __host__ void free()
    {
        CudaMemoryManager::getInstance().freeMemory(_selection);
    }

    __host__ SelectedEntitites getSelection()
    {
        SelectedEntitites result;
        checkCudaErrors(cudaMemcpy(&result, _selection, sizeof(SelectedEntitites), cudaMemcpyDeviceToHost));
        return result;
    }

    __device__ void reset()
    {
        _selection->numCells = 0;
        _selection->numIndirectCells = 0;
        _selection->numParticles = 0;
    }

    __device__ void incSelectedCell()
    {
        atomicAdd(&_selection->numCells, 1);
        atomicAdd(&_selection->numIndirectCells, 1);
    }

    __device__ void incIndirectSelectedCell()
    {
        atomicAdd(&_selection->numIndirectCells, 1);
    }

    __device__ void incSelectedParticle()
    {
        atomicAdd(&_selection->numParticles, 1);
    }

private:
    SelectedEntitites* _selection;
};