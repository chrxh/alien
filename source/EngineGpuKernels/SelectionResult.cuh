#pragma once

#include "EngineInterface/SelectionShallowData.h"
#include "Definitions.cuh"
#include "Cell.cuh"

class SelectionResult
{
public:
    __host__ void init()
    {
        CudaMemoryManager::getInstance().acquireMemory<SelectionShallowData>(1, _selectionShallowData);
        CHECK_FOR_CUDA_ERROR(cudaMemset(_selectionShallowData, 0, sizeof(SelectionShallowData)));
    }

    __host__ void free()
    {
        CudaMemoryManager::getInstance().freeMemory(_selectionShallowData);
    }

    __host__ SelectionShallowData getSelectionShallowData()
    {
        SelectionShallowData result;
        CHECK_FOR_CUDA_ERROR(
            cudaMemcpy(&result, _selectionShallowData, sizeof(SelectionShallowData), cudaMemcpyDeviceToHost));

        return result;
    }

    __device__ void reset()
    {
        _selectionShallowData->numCells = 0;
        _selectionShallowData->numClusterCells = 0;
        _selectionShallowData->numParticles = 0;

        _selectionShallowData->centerPosX = 0;
        _selectionShallowData->centerPosY = 0;
        _selectionShallowData->centerVelX = 0;
        _selectionShallowData->centerVelY = 0;

        _selectionShallowData->extCenterPosX = 0;
        _selectionShallowData->extCenterPosY = 0;
        _selectionShallowData->extCenterVelX = 0;
        _selectionShallowData->extCenterVelY = 0;
    }

    __device__ void collectCell(Cell* cell)
    {
        if (1 == cell->selected) {
            atomicAdd(&_selectionShallowData->numCells, 1);
            atomicAdd(&_selectionShallowData->centerPosX, cell->absPos.x);
            atomicAdd(&_selectionShallowData->centerPosY, cell->absPos.y);
            atomicAdd(&_selectionShallowData->centerVelX, cell->vel.x);
            atomicAdd(&_selectionShallowData->centerVelY, cell->vel.y);
        }

        atomicAdd(&_selectionShallowData->numClusterCells, 1);
        atomicAdd(&_selectionShallowData->extCenterPosX, cell->absPos.x);
        atomicAdd(&_selectionShallowData->extCenterPosY, cell->absPos.y);
        atomicAdd(&_selectionShallowData->extCenterVelX, cell->vel.x);
        atomicAdd(&_selectionShallowData->extCenterVelY, cell->vel.y);
    }

    __device__ void collectParticle(Particle* particle)
    {
        atomicAdd(&_selectionShallowData->numParticles, 1);
        atomicAdd(&_selectionShallowData->centerPosX, particle->absPos.x);
        atomicAdd(&_selectionShallowData->centerPosY, particle->absPos.y);
        atomicAdd(&_selectionShallowData->centerVelX, particle->vel.x);
        atomicAdd(&_selectionShallowData->centerVelY, particle->vel.y);
        atomicAdd(&_selectionShallowData->extCenterPosX, particle->absPos.x);
        atomicAdd(&_selectionShallowData->extCenterPosY, particle->absPos.y);
        atomicAdd(&_selectionShallowData->extCenterVelX, particle->vel.x);
        atomicAdd(&_selectionShallowData->extCenterVelY, particle->vel.y);
    }

    __device__ void finalize()
    {
        auto numEntities = _selectionShallowData->numCells + _selectionShallowData->numParticles;
        if (numEntities > 0) {
            _selectionShallowData->centerPosX /= numEntities;
            _selectionShallowData->centerPosY /= numEntities;
            _selectionShallowData->centerVelX /= numEntities;
            _selectionShallowData->centerVelY /= numEntities;
        }

        auto numExtEntities = _selectionShallowData->numClusterCells + _selectionShallowData->numParticles;
        if (numEntities > 0) {
            _selectionShallowData->extCenterPosX /= numExtEntities;
            _selectionShallowData->extCenterPosY /= numExtEntities;
            _selectionShallowData->extCenterVelX /= numExtEntities;
            _selectionShallowData->extCenterVelY /= numExtEntities;
        }
    }

private:
    SelectionShallowData* _selectionShallowData;
};