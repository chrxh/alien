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

    __device__ __inline__ uint32_t getDensity(float2 const& pos, int color)
    {
        auto index = toInt(pos.x) / _slotSize + toInt(pos.y) / _slotSize * _densityMapSize.x;
        if (index >= 0 && index < _densityMapSize.x * _densityMapSize.y) {
            return (_densityMap[index] >> (color * 8)) & 0xff;
        }
        return 0;
    }

    __device__ __inline__ void addCell(Cell* cell)
    {
        auto index = toInt(cell->absPos.x) / _slotSize + toInt(cell->absPos.y) / _slotSize * _densityMapSize.x;
        if (index >= 0 && index < _densityMapSize.x * _densityMapSize.y) {
            auto color = calcMod(cell->metadata.color, 7);
            alienAtomicAdd(&_densityMap[index], uint64_t(1) << (color * 8));
        }
    }

private:

    int _slotSize;
    int2 _densityMapSize;
    uint64_t* _densityMap;
};

