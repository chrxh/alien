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
        CudaMemoryManager::getInstance().acquireMemory<uint64_t>(_densityMapSize.x * _densityMapSize.y, _otherMutantDensityMap);
        _slotSize = slotSize;
    }

    __host__ __inline__ void free()
    {
        CudaMemoryManager::getInstance().freeMemory(_colorDensityMap);
        CudaMemoryManager::getInstance().freeMemory(_otherMutantDensityMap);
    }

    __device__ __inline__ void clear()
    {
        auto const partition = calcAllThreadsPartition(_densityMapSize.x * _densityMapSize.y);
        for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
            _colorDensityMap[index] = 0;
            _otherMutantDensityMap[index] = 0;
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

    __device__ __inline__ uint32_t getOtherMutantsDensity(uint64_t const& timestep, float2 const& pos, uint32_t mutationId) const
    {
        auto index = toInt(pos.x) / _slotSize + toInt(pos.y) / _slotSize * _densityMapSize.x;
        if (index >= 0 && index < _densityMapSize.x * _densityMapSize.y) {
            uint64_t bucket = calcBucket(mutationId, timestep);
            return (_otherMutantDensityMap[index] >> (bucket * 8)) & 0xff;
        }
        return 0;
    }

    __device__ __inline__ void addCell(uint64_t const& timestep, Cell* cell)
    {
        auto index = toInt(cell->pos.x) / _slotSize + toInt(cell->pos.y) / _slotSize * _densityMapSize.x;
        if (index >= 0 && index < _densityMapSize.x * _densityMapSize.y) {
            auto color = calcMod(cell->color, MAX_COLORS);
            alienAtomicAdd64(&_colorDensityMap[index], (1ull << (color * 8)) | (1ull << 56));

            if (cell->mutationId != 0) {
                uint64_t bucket = calcBucket(cell->mutationId, timestep);
                alienAtomicAdd64(&_otherMutantDensityMap[index], 0x0101010101010101ull ^ (1ull << (bucket * 8)));
            } else {
                alienAtomicAdd64(&_otherMutantDensityMap[index], 1ull);
            }
        }
    }

private:
    // timestep is used as an offset to avoid same buckets for different mutationIds for all times
    __device__ __inline__ uint64_t calcBucket(uint32_t const& mutationId, uint64_t const& timestep) const
    {
        return mutationId != 0 ? (static_cast<uint64_t>(mutationId) + timestep / 23) % 8 : 0;
    }

    int _slotSize;
    int2 _densityMapSize;
    uint64_t* _colorDensityMap;
    uint64_t* _otherMutantDensityMap;
};

