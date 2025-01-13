﻿#pragma once

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
        CudaMemoryManager::getInstance().acquireMemory<uint64_t>(_densityMapSize.x * _densityMapSize.y, _sameMutantDensityMap1);
        CudaMemoryManager::getInstance().acquireMemory<uint64_t>(_densityMapSize.x * _densityMapSize.y, _sameMutantDensityMap2);
        CudaMemoryManager::getInstance().acquireMemory<uint32_t>(_densityMapSize.x * _densityMapSize.y, _specificCellTypeDensityMap);
        CudaMemoryManager::getInstance().acquireMemory<uint64_t>(_densityMapSize.x * _densityMapSize.y, _lessGenomeComplexityDensityMap1);
        CudaMemoryManager::getInstance().acquireMemory<uint64_t>(_densityMapSize.x * _densityMapSize.y, _lessGenomeComplexityDensityMap2);
        CudaMemoryManager::getInstance().acquireMemory<uint64_t>(_densityMapSize.x * _densityMapSize.y, _moreGenomeComplexityDensityMap1);
        CudaMemoryManager::getInstance().acquireMemory<uint64_t>(_densityMapSize.x * _densityMapSize.y, _moreGenomeComplexityDensityMap2);
        _slotSize = slotSize;
    }

    __host__ __inline__ void free()
    {
        CudaMemoryManager::getInstance().freeMemory(_colorDensityMap);
        CudaMemoryManager::getInstance().freeMemory(_otherMutantDensityMap);
        CudaMemoryManager::getInstance().freeMemory(_sameMutantDensityMap1);
        CudaMemoryManager::getInstance().freeMemory(_sameMutantDensityMap2);
        CudaMemoryManager::getInstance().freeMemory(_specificCellTypeDensityMap);
        CudaMemoryManager::getInstance().freeMemory(_lessGenomeComplexityDensityMap1);
        CudaMemoryManager::getInstance().freeMemory(_lessGenomeComplexityDensityMap2);
        CudaMemoryManager::getInstance().freeMemory(_moreGenomeComplexityDensityMap1);
        CudaMemoryManager::getInstance().freeMemory(_moreGenomeComplexityDensityMap2);
    }

    __device__ __inline__ void clear()
    {
        auto const partition = calcAllThreadsPartition(_densityMapSize.x * _densityMapSize.y);
        for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
            _colorDensityMap[index] = 0;
            _otherMutantDensityMap[index] = 0;
            _sameMutantDensityMap1[index] = 0;
            _sameMutantDensityMap2[index] = 0;
            _specificCellTypeDensityMap[index] = 0;
            _lessGenomeComplexityDensityMap1[index] = 0;
            _lessGenomeComplexityDensityMap2[index] = 0;
            _moreGenomeComplexityDensityMap1[index] = 0;
            _moreGenomeComplexityDensityMap2[index] = 0;
        }
    }

    __device__ __inline__ uint32_t getCellDensity(float2 const& pos) const
    {
        auto index = toInt(pos.x) / _slotSize + toInt(pos.y) / _slotSize * _densityMapSize.x;
        if (index >= 0 && index < _densityMapSize.x * _densityMapSize.y) {
            return static_cast<uint32_t>((_colorDensityMap[index] >> 56) & 0xff);
        }
        return 0;
    }

    __device__ __inline__ uint32_t getColorDensity(float2 const& pos, int color) const
    {
        auto index = toInt(pos.x) / _slotSize + toInt(pos.y) / _slotSize * _densityMapSize.x;
        if (index >= 0 && index < _densityMapSize.x * _densityMapSize.y) {
            return static_cast<uint32_t>((_colorDensityMap[index] >> (color * 8)) & 0xff);
        }
        return 0;
    }

    __device__ __inline__ uint32_t getOtherMutantDensity(uint64_t const& timestep, float2 const& pos, uint32_t mutationId) const
    {
        auto index = toInt(pos.x) / _slotSize + toInt(pos.y) / _slotSize * _densityMapSize.x;
        if (index >= 0 && index < _densityMapSize.x * _densityMapSize.y) {
            uint64_t bucket = calcOtherMutantsBucket(mutationId, timestep);
            return static_cast<uint32_t>((_otherMutantDensityMap[index] >> (bucket * 8)) & 0xff);
        }
        return 0ul;
    }

    __device__ __inline__ uint32_t getSameMutantDensity(float2 const& pos, uint32_t mutationId) const
    {
        auto index = toInt(pos.x) / _slotSize + toInt(pos.y) / _slotSize * _densityMapSize.x;
        if (index >= 0 && index < _densityMapSize.x * _densityMapSize.y) {
            uint64_t bucket1 = mutationId % 3;
            uint64_t bucket2 = mutationId % 5;
            uint64_t bucket3 = mutationId % 7;
            auto densityMapEntry = _sameMutantDensityMap1[index];
            auto density1 = (densityMapEntry >> (bucket1 * 8)) & 0xff;
            auto density2 = (densityMapEntry >> ((bucket2 + 3) * 8)) & 0xff;
            auto density3 = (_sameMutantDensityMap2[index] >> (bucket3 * 8)) & 0xff;
            return static_cast<uint32_t>(min(min(density1, density2), density3));
        }
        return 0ul;
    }

    __device__ __inline__ uint32_t getFreeCellDensity(float2 const& pos) const
    {
        auto index = toInt(pos.x) / _slotSize + toInt(pos.y) / _slotSize * _densityMapSize.x;
        if (index >= 0 && index < _densityMapSize.x * _densityMapSize.y) {
            return (_specificCellTypeDensityMap[index] >> 8) & 0xff;
        }
        return 0ul;
    }

    __device__ __inline__ uint32_t getStructureDensity(float2 const& pos) const
    {
        auto index = toInt(pos.x) / _slotSize + toInt(pos.y) / _slotSize * _densityMapSize.x;
        if (index >= 0 && index < _densityMapSize.x * _densityMapSize.y) {
            return _specificCellTypeDensityMap[index] & 0xff;
        }
        return 0ul;
    }

    __device__ __inline__ uint32_t getLessComplexMutantDensity(float2 const& pos, float genomeComplexity) const
    {
        auto index = toInt(pos.x) / _slotSize + toInt(pos.y) / _slotSize * _densityMapSize.x;
        if (index >= 0 && index < _densityMapSize.x * _densityMapSize.y) {
            auto bucket = min(16, max(0, 31 - __clz(convertGenomeComplexityToIntValue(genomeComplexity))));
            if (bucket < 8) {
                return (_lessGenomeComplexityDensityMap1[index] >> (bucket * 8)) & 0xff;
            } else {
                return (_lessGenomeComplexityDensityMap2[index] >> ((bucket - 8) * 8)) & 0xff;
            }
        }
        return 0ul;
    }

    __device__ __inline__ uint32_t getMoreComplexMutantDensity(float2 const& pos, float genomeComplexity) const
    {
        auto index = toInt(pos.x) / _slotSize + toInt(pos.y) / _slotSize * _densityMapSize.x;
        if (index >= 0 && index < _densityMapSize.x * _densityMapSize.y) {
            auto bucket = min(16, max(0, 33 - __clz(convertGenomeComplexityToIntValue(genomeComplexity))));
            if (bucket < 8) {
                return (_moreGenomeComplexityDensityMap1[index] >> (bucket * 8)) & 0xff;
            } else {
                return (_moreGenomeComplexityDensityMap2[index] >> ((bucket - 8) * 8)) & 0xff;
            }
        }
        return 0ul;
    }

    __device__ __inline__ void addCell(uint64_t const& timestep, Cell* cell)
    {
        auto index = toInt(cell->pos.x) / _slotSize + toInt(cell->pos.y) / _slotSize * _densityMapSize.x;
        if (index >= 0 && index < _densityMapSize.x * _densityMapSize.y) {
            auto color = calcMod(cell->color, MAX_COLORS);
            alienAtomicAdd64(&_colorDensityMap[index], static_cast<uint64_t>((1ull << (color * 8)) | (1ull << 56)));

            if (cell->cellType == CellType_Structure) {
                alienAtomicAdd32(&_specificCellTypeDensityMap[index], static_cast<uint32_t>(1));
            } else if (cell->cellType == CellType_Free) {
                alienAtomicAdd32(&_specificCellTypeDensityMap[index], static_cast<uint32_t>(0x100));
            } else {
                {
                    uint64_t bucket = calcOtherMutantsBucket(cell->mutationId, timestep);
                    alienAtomicAdd64(&_otherMutantDensityMap[index], static_cast<uint64_t>(0x0101010101010101ull ^ (1ull << (bucket * 8))));
                }
                {
                    uint64_t bucket1 = cell->mutationId % 3;
                    uint64_t bucket2 = cell->mutationId % 5;
                    uint64_t bucket3 = cell->mutationId % 7;
                    alienAtomicAdd64(&_sameMutantDensityMap1[index], static_cast<uint64_t>((1ull << (bucket1 * 8)) | (1ull << ((bucket2 + 3) * 8))));
                    alienAtomicAdd64(&_sameMutantDensityMap2[index], static_cast<uint64_t>(1ull << (bucket3 * 8)));
                }
                {
                    auto bucket = 32 - __clz(convertGenomeComplexityToIntValue(cell->genomeComplexity));
                    if (bucket < 8) {
                        {
                            auto bitset = static_cast<uint64_t>(1ull << bucket * 8);
                            for (int i = 0; i < 7; ++i) {
                                bitset |= (bitset << 8);
                            }
                            alienAtomicAdd64(&_lessGenomeComplexityDensityMap1[index], bitset);
                            alienAtomicAdd64(&_lessGenomeComplexityDensityMap2[index], static_cast<uint64_t>(0x0101010101010101ull));
                        }
                        {
                            auto bitset = static_cast<uint64_t>(1ull << bucket * 8);
                            for (int i = 0; i < 7; ++i) {
                                bitset |= (bitset >> 8);
                            }
                            alienAtomicAdd64(&_moreGenomeComplexityDensityMap1[index], bitset);
                        }
                    } else if (bucket < 16) {
                        {
                            auto bitset = static_cast<uint64_t>(1ull << ((bucket - 8) * 8));
                            for (int i = 0; i < 7; ++i) {
                                bitset |= (bitset << 8);
                            }
                            alienAtomicAdd64(&_lessGenomeComplexityDensityMap2[index], bitset);
                        }
                        {
                            auto bitset = static_cast<uint64_t>(1ull << ((bucket - 8) * 8));
                            for (int i = 0; i < 7; ++i) {
                                bitset |= (bitset >> 8);
                            }
                            alienAtomicAdd64(&_moreGenomeComplexityDensityMap2[index], bitset);
                            alienAtomicAdd64(&_moreGenomeComplexityDensityMap1[index], static_cast<uint64_t>(0x0101010101010101ull));
                        }
                    }
                }
            }
        }
    }

private:
    // timestep is used as an offset to avoid same buckets for different mutationIds for all times
    __device__ __inline__ uint64_t calcOtherMutantsBucket(uint32_t const& mutationId, uint64_t const& timestep) const
    {
        return mutationId != 0 ? (static_cast<uint64_t>(mutationId) + timestep / 23) % 8 : 0;
    }

    __device__ __inline__ uint32_t convertGenomeComplexityToIntValue(float genomeComplexity) const { return toInt(genomeComplexity * 10); }

    int _slotSize;
    int2 _densityMapSize;
    uint64_t* _colorDensityMap;
    uint64_t* _otherMutantDensityMap;
    uint64_t* _sameMutantDensityMap1;
    uint64_t* _sameMutantDensityMap2;
    uint32_t* _respawnedMutantDensityMap;
    uint32_t* _specificCellTypeDensityMap;
    uint64_t* _lessGenomeComplexityDensityMap1;
    uint64_t* _lessGenomeComplexityDensityMap2;
    uint64_t* _moreGenomeComplexityDensityMap1;
    uint64_t* _moreGenomeComplexityDensityMap2;
};

