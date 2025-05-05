﻿#pragma once

#include "EngineInterface/SelectionShallowData.h"
#include "Definitions.cuh"
#include "Object.cuh"

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

        _selectionShallowData->clusterCenterPosX = 0;
        _selectionShallowData->clusterCenterPosY = 0;
        _selectionShallowData->clusterCenterVelX = 0;
        _selectionShallowData->clusterCenterVelY = 0;
    }

    __device__ void collectCell(Cell* cell, float2 refPos, BaseMap const& map)
    {
        auto pos = cell->pos + map.getCorrectionIncrement(refPos, cell->pos);

        if (1 == cell->selected) {
            atomicAdd(&_selectionShallowData->numCells, 1);
            atomicAdd(&_selectionShallowData->centerPosX, pos.x);
            atomicAdd(&_selectionShallowData->centerPosY, pos.y);
            atomicAdd(&_selectionShallowData->centerVelX, cell->vel.x);
            atomicAdd(&_selectionShallowData->centerVelY, cell->vel.y);
        }

        atomicAdd(&_selectionShallowData->numClusterCells, 1);
        atomicAdd(&_selectionShallowData->clusterCenterPosX, pos.x);
        atomicAdd(&_selectionShallowData->clusterCenterPosY, pos.y);
        atomicAdd(&_selectionShallowData->clusterCenterVelX, cell->vel.x);
        atomicAdd(&_selectionShallowData->clusterCenterVelY, cell->vel.y);
    }

    __device__ void collectParticle(Particle* particle, float2 refPos, BaseMap const& map)
    {
        auto pos = particle->pos + map.getCorrectionIncrement(refPos, particle->pos);

        atomicAdd(&_selectionShallowData->numParticles, 1);
        atomicAdd(&_selectionShallowData->centerPosX, pos.x);
        atomicAdd(&_selectionShallowData->centerPosY, pos.y);
        atomicAdd(&_selectionShallowData->centerVelX, particle->vel.x);
        atomicAdd(&_selectionShallowData->centerVelY, particle->vel.y);
        atomicAdd(&_selectionShallowData->clusterCenterPosX, pos.x);
        atomicAdd(&_selectionShallowData->clusterCenterPosY, pos.y);
        atomicAdd(&_selectionShallowData->clusterCenterVelX, particle->vel.x);
        atomicAdd(&_selectionShallowData->clusterCenterVelY, particle->vel.y);
    }

    __device__ void finalize(BaseMap const& map, bool mapCorrection)
    {
        auto numEntities = _selectionShallowData->numCells + _selectionShallowData->numParticles;
        if (numEntities > 0) {
            _selectionShallowData->centerPosX /= numEntities;
            _selectionShallowData->centerPosY /= numEntities;
            _selectionShallowData->centerVelX /= numEntities;
            _selectionShallowData->centerVelY /= numEntities;
            if (mapCorrection) {
                auto correctedPos = map.getCorrectedPosition({_selectionShallowData->centerPosX, _selectionShallowData->centerPosY});
                _selectionShallowData->centerPosX = correctedPos.x;
                _selectionShallowData->centerPosY = correctedPos.y;
            }
        }

        auto numExtEntities = _selectionShallowData->numClusterCells + _selectionShallowData->numParticles;
        if (numEntities > 0) {
            _selectionShallowData->clusterCenterPosX /= numExtEntities;
            _selectionShallowData->clusterCenterPosY /= numExtEntities;
            _selectionShallowData->clusterCenterVelX /= numExtEntities;
            _selectionShallowData->clusterCenterVelY /= numExtEntities;
            if (mapCorrection) {
                auto correctedPos = map.getCorrectedPosition({_selectionShallowData->clusterCenterPosX, _selectionShallowData->clusterCenterPosY});
                _selectionShallowData->clusterCenterPosX = correctedPos.x;
                _selectionShallowData->clusterCenterPosY = correctedPos.y;
            }
        }
    }

private:
    SelectionShallowData* _selectionShallowData;
};