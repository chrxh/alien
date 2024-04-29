#pragma once

#include "Base.cuh"
#include "CudaMemoryManager.cuh"
#include "Array.cuh"
#include "Object.cuh"

class DensityMap
{
public:
    __host__ __inline__ void init(int2 const& worldSize, int slotSize)
    {
        _densityMapSize = {worldSize.x / slotSize, worldSize.y / slotSize};
        CudaMemoryManager::getInstance().acquireMemory<uint64_t>(_densityMapSize.x * _densityMapSize.y, _colorDensityMap);
        CudaMemoryManager::getInstance().acquireMemory<uint64_t>(_densityMapSize.x * _densityMapSize.y, _mutantDensityMap);
        _slotSize = slotSize;
    }

    __host__ __inline__ void free()
    {
        CudaMemoryManager::getInstance().freeMemory(_colorDensityMap);
        CudaMemoryManager::getInstance().freeMemory(_mutantDensityMap);
    }

    __device__ __inline__ void clear()
    {
        auto const partition = calcAllThreadsPartition(_densityMapSize.x * _densityMapSize.y);
        for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
            _colorDensityMap[index] = 0;
            _mutantDensityMap[index] = 0;
        }
    }

    __device__ __inline__ uint32_t getCellDensity(float2 const& pos) const
    {
        auto index = toInt(pos.x) / _slotSize + toInt(pos.y) / _slotSize * _densityMapSize.x;
        if (index >= 0 && index < _densityMapSize.x * _densityMapSize.y) {
            return (_colorDensityMap[index] >> 56) & 0xff;
        }
        return 0;
    }

    __device__ __inline__ uint32_t getColorDensity(float2 const& pos, int color) const
    {
        auto index = toInt(pos.x) / _slotSize + toInt(pos.y) / _slotSize * _densityMapSize.x;
        if (index >= 0 && index < _densityMapSize.x * _densityMapSize.y) {
            return (_colorDensityMap[index] >> (color * 8)) & 0xff;
        }
        return 0;
    }

    __device__ __inline__ uint32_t getOtherMutantsDensity(float2 const& pos, uint64_t mutationId) const
    {
        auto index = toInt(pos.x) / _slotSize + toInt(pos.y) / _slotSize * _densityMapSize.x;
        if (index >= 0 && index < _densityMapSize.x * _densityMapSize.y) {
            auto bucket = mutationId % 8;
            return (_mutantDensityMap[index] >> (bucket * 8)) & 0xff;
        }
        return 0;
    }

    __device__ __inline__ void addCell(Cell* cell)
    {
        auto index = toInt(cell->pos.x) / _slotSize + toInt(cell->pos.y) / _slotSize * _densityMapSize.x;
        if (index >= 0 && index < _densityMapSize.x * _densityMapSize.y) {
            auto color = calcMod(cell->color, MAX_COLORS);
            alienAtomicAdd64(&_colorDensityMap[index], (uint64_t(1) << (color * 8)) | (uint64_t(1) << 56));

            if (cell->mutationId != 0) {
                auto bucket = cell->mutationId % 8;
                alienAtomicAdd64(&_mutantDensityMap[index], 0x0101010101010101ull ^ (1 << (bucket * 8)));
            } else {
                alienAtomicAdd64(&_mutantDensityMap[index], 1ull);
            }
        }
    }

private:
    int _slotSize;
    int2 _densityMapSize;
    uint64_t* _colorDensityMap;
    uint64_t* _mutantDensityMap;
};

