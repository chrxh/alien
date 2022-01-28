#pragma once

#include "Base.cuh"
#include "CudaMemoryManager.cuh"
#include "Array.cuh"
#include "Cell.cuh"

class DensityMap
{
public:
    __host__ __inline__ void init(int2 const& worldSize, int slotSize)
    {
        _densityMapSize = {worldSize.x / slotSize, worldSize.y / slotSize};
        CudaMemoryManager::getInstance().acquireMemory<uint64_t>(_densityMapSize.x * _densityMapSize.y, _densityMap);
        _slotSize = slotSize;
    }

    __host__ __inline__ void free() { CudaMemoryManager::getInstance().freeMemory(_densityMap); }

    __device__ __inline__ void clear()
    {
        auto const partition = calcAllThreadsPartition(_densityMapSize.x * _densityMapSize.y);
        for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
            _densityMap[index] = 0;
        }
    }

    __device__ __inline__ void createMap(Array<Cell*> cells)
    {
        auto const partition = calcAllThreadsPartition(cells.getNumEntries());
        for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
            addCell(cells.at(index)->absPos);
        }
    }

    __device__ __inline__ uint32_t getDensity(float2 const& pos)
    {
        auto index = toInt(pos.x) / _slotSize + toInt(pos.y) / _slotSize * _densityMapSize.x;
        if (index >= 0 && index < _densityMapSize.x * _densityMapSize.y) {
            return _densityMap[index];
        }
        return 0;
    }

private:
    __device__ __inline__ void addCell(float2 const& pos)
    {
        auto index = toInt(pos.x) / _slotSize + toInt(pos.y) / _slotSize * _densityMapSize.x;
        if (index >= 0 && index < _densityMapSize.x * _densityMapSize.y) {
            atomicAdd(&_densityMap[index], 1);
        }
    }

    int _slotSize;
    int2 _densityMapSize;
    uint64_t* _densityMap;
};

