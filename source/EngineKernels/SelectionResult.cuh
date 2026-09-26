#pragma once

#include <EngineInterface/SelectionShallowData.h>

#include "Entities.cuh"

class SelectionResult
{
public:
    __host__ void init()
    {
        CudaMemoryManager::getInstance().acquireMemory<SelectionShallowData>(1, _selectionShallowData);
        CHECK_FOR_DEVICE_ERRORS(cudaMemset(_selectionShallowData, 0, sizeof(SelectionShallowData)));
    }

    __host__ void free() { CudaMemoryManager::getInstance().freeMemory(_selectionShallowData); }

    __host__ SelectionShallowData getSelectionShallowData()
    {
        SelectionShallowData result;
        CHECK_FOR_DEVICE_ERRORS(cudaMemcpy(&result, _selectionShallowData, sizeof(SelectionShallowData), cudaMemcpyDeviceToHost));

        return result;
    }

    __device__ void reset()
    {
        _selectionShallowData->numObjects = 0;
        _selectionShallowData->numCreatures = 0;
        _selectionShallowData->numClusterCells = 0;
        _selectionShallowData->numEnergyParticles = 0;

        _selectionShallowData->centerPosX = 0;
        _selectionShallowData->centerPosY = 0;
        _selectionShallowData->centerVelX = 0;
        _selectionShallowData->centerVelY = 0;

        _selectionShallowData->clusterCenterPosX = 0;
        _selectionShallowData->clusterCenterPosY = 0;
        _selectionShallowData->clusterCenterVelX = 0;
        _selectionShallowData->clusterCenterVelY = 0;

        _selectionShallowData->minPosX = Infinity<float>::value;
        _selectionShallowData->minPosY = Infinity<float>::value;
        _selectionShallowData->maxPosX = -Infinity<float>::value;
        _selectionShallowData->maxPosY = -Infinity<float>::value;
        _selectionShallowData->clusterMinPosX = Infinity<float>::value;
        _selectionShallowData->clusterMinPosY = Infinity<float>::value;
        _selectionShallowData->clusterMaxPosX = -Infinity<float>::value;
        _selectionShallowData->clusterMaxPosY = -Infinity<float>::value;
    }

    __device__ void collectObject(Object* object, float2 pos)
    {
        if (1 == object->selected) {
            atomicAdd(&_selectionShallowData->numObjects, 1);
            atomicAdd(&_selectionShallowData->centerPosX, pos.x);
            atomicAdd(&_selectionShallowData->centerPosY, pos.y);
            atomicAdd(&_selectionShallowData->centerVelX, object->vel.x);
            atomicAdd(&_selectionShallowData->centerVelY, object->vel.y);
            collectBounds(pos);
        }

        atomicAdd(&_selectionShallowData->numClusterCells, 1);
        atomicAdd(&_selectionShallowData->clusterCenterPosX, pos.x);
        atomicAdd(&_selectionShallowData->clusterCenterPosY, pos.y);
        atomicAdd(&_selectionShallowData->clusterCenterVelX, object->vel.x);
        atomicAdd(&_selectionShallowData->clusterCenterVelY, object->vel.y);
        collectClusterBounds(pos);
    }

    __device__ void collectCreature() { atomicAdd(&_selectionShallowData->numCreatures, 1); }

    __device__ void collectParticle(Energy* particle, float2 pos)
    {
        atomicAdd(&_selectionShallowData->numEnergyParticles, 1);
        atomicAdd(&_selectionShallowData->centerPosX, pos.x);
        atomicAdd(&_selectionShallowData->centerPosY, pos.y);
        atomicAdd(&_selectionShallowData->centerVelX, particle->vel.x);
        atomicAdd(&_selectionShallowData->centerVelY, particle->vel.y);
        atomicAdd(&_selectionShallowData->clusterCenterPosX, pos.x);
        atomicAdd(&_selectionShallowData->clusterCenterPosY, pos.y);
        atomicAdd(&_selectionShallowData->clusterCenterVelX, particle->vel.x);
        atomicAdd(&_selectionShallowData->clusterCenterVelY, particle->vel.y);
        collectBounds(pos);
        collectClusterBounds(pos);
    }

    __device__ void finalize(BaseMap const& map, bool mapCorrection)
    {
        auto numEntities = _selectionShallowData->numObjects + _selectionShallowData->numEnergyParticles;
        if (numEntities > 0) {
            _selectionShallowData->centerPosX /= numEntities;
            _selectionShallowData->centerPosY /= numEntities;
            _selectionShallowData->centerVelX /= numEntities;
            _selectionShallowData->centerVelY /= numEntities;
            if (mapCorrection) {
                auto correctedPos = map.getCorrectedPosition({_selectionShallowData->centerPosX, _selectionShallowData->centerPosY});
                auto correction = correctedPos - float2{_selectionShallowData->centerPosX, _selectionShallowData->centerPosY};
                _selectionShallowData->centerPosX = correctedPos.x;
                _selectionShallowData->centerPosY = correctedPos.y;
                _selectionShallowData->minPosX += correction.x;
                _selectionShallowData->minPosY += correction.y;
                _selectionShallowData->maxPosX += correction.x;
                _selectionShallowData->maxPosY += correction.y;
            }
        } else {
            _selectionShallowData->minPosX = 0;
            _selectionShallowData->minPosY = 0;
            _selectionShallowData->maxPosX = 0;
            _selectionShallowData->maxPosY = 0;
        }

        auto numExtEntities = _selectionShallowData->numClusterCells + _selectionShallowData->numEnergyParticles;
        if (numExtEntities > 0) {
            _selectionShallowData->clusterCenterPosX /= numExtEntities;
            _selectionShallowData->clusterCenterPosY /= numExtEntities;
            _selectionShallowData->clusterCenterVelX /= numExtEntities;
            _selectionShallowData->clusterCenterVelY /= numExtEntities;
            if (mapCorrection) {
                auto correctedPos = map.getCorrectedPosition({_selectionShallowData->clusterCenterPosX, _selectionShallowData->clusterCenterPosY});
                auto correction = correctedPos - float2{_selectionShallowData->clusterCenterPosX, _selectionShallowData->clusterCenterPosY};
                _selectionShallowData->clusterCenterPosX = correctedPos.x;
                _selectionShallowData->clusterCenterPosY = correctedPos.y;
                _selectionShallowData->clusterMinPosX += correction.x;
                _selectionShallowData->clusterMinPosY += correction.y;
                _selectionShallowData->clusterMaxPosX += correction.x;
                _selectionShallowData->clusterMaxPosY += correction.y;
            }
        } else {
            _selectionShallowData->clusterMinPosX = 0;
            _selectionShallowData->clusterMinPosY = 0;
            _selectionShallowData->clusterMaxPosX = 0;
            _selectionShallowData->clusterMaxPosY = 0;
        }
    }

private:
    __device__ void collectBounds(float2 const& pos)
    {
        alienAtomicMin(&_selectionShallowData->minPosX, pos.x);
        alienAtomicMin(&_selectionShallowData->minPosY, pos.y);
        alienAtomicMax(&_selectionShallowData->maxPosX, pos.x);
        alienAtomicMax(&_selectionShallowData->maxPosY, pos.y);
    }

    __device__ void collectClusterBounds(float2 const& pos)
    {
        alienAtomicMin(&_selectionShallowData->clusterMinPosX, pos.x);
        alienAtomicMin(&_selectionShallowData->clusterMinPosY, pos.y);
        alienAtomicMax(&_selectionShallowData->clusterMaxPosX, pos.x);
        alienAtomicMax(&_selectionShallowData->clusterMaxPosY, pos.y);
    }

    SelectionShallowData* _selectionShallowData;
};