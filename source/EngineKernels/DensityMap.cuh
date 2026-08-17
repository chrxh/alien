#pragma once

#include "Entities.cuh"

class DensityMap
{
public:
    __host__ __inline__ void init(int2 const& worldSize, int slotSize)
    {
        _densityMapSize = {worldSize.x / slotSize, worldSize.y / slotSize};
        CudaMemoryManager::getInstance().acquireMemory<float>(_densityMapSize.x * _densityMapSize.y, _energyParticleDensityMap);
        CudaMemoryManager::getInstance().acquireMemory<uint64_t>(_densityMapSize.x * _densityMapSize.y, _freeCellDensityMap1);
        CudaMemoryManager::getInstance().acquireMemory<uint64_t>(_densityMapSize.x * _densityMapSize.y, _freeCellDensityMap2);
        CudaMemoryManager::getInstance().acquireMemory<uint32_t>(_densityMapSize.x * _densityMapSize.y, _solidDensityMap);
        _slotSize = slotSize;
    }

    __host__ __inline__ void free()
    {
        CudaMemoryManager::getInstance().freeMemory(_energyParticleDensityMap);
        CudaMemoryManager::getInstance().freeMemory(_freeCellDensityMap1);
        CudaMemoryManager::getInstance().freeMemory(_freeCellDensityMap2);
        CudaMemoryManager::getInstance().freeMemory(_solidDensityMap);
    }

    __device__ __inline__ void clear()
    {
        auto const partition = calcSystemThreadPartition(_densityMapSize.x * _densityMapSize.y);
        for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
            _energyParticleDensityMap[index] = 0.0f;
            _freeCellDensityMap1[index] = 0;
            _freeCellDensityMap2[index] = 0;
            _solidDensityMap[index] = 0;
        }
    }

    __device__ __inline__ float getEnergyParticleDensity(float2 const& pos) const
    {
        auto index = toInt(pos.x) / _slotSize + toInt(pos.y) / _slotSize * _densityMapSize.x;
        if (index >= 0 && index < _densityMapSize.x * _densityMapSize.y) {
            auto slotSizeAsFlot = toFloat(_slotSize);
            return _energyParticleDensityMap[index] / (slotSizeAsFlot * slotSizeAsFlot);
        }
        return 0.0f;
    }

    __device__ __inline__ float getFreeCellDensity(float2 const& pos, uint16_t restrictToColors) const
    {
        auto index = toInt(pos.x) / _slotSize + toInt(pos.y) / _slotSize * _densityMapSize.x;
        if (index >= 0 && index < _densityMapSize.x * _densityMapSize.y) {
            auto slotSizeAsFlot = toFloat(_slotSize);
            if (restrictToColors == 0x3ff) {
                auto totalCount = (_freeCellDensityMap2[index] >> 16) & 0xff;
                return toFloat(totalCount) / (slotSizeAsFlot * slotSizeAsFlot);
            } else {
                int matchedColorCount = 0;
                for (int color = 0; color < MAX_COLORS; ++color) {
                    if ((restrictToColors >> color) & 1) {
                        if (color < 8) {
                            matchedColorCount += (_freeCellDensityMap1[index] >> (color * 8)) & 0xff;
                        } else {
                            matchedColorCount += (_freeCellDensityMap2[index] >> ((color - 8) * 8)) & 0xff;
                        }
                    }
                }
                return toFloat(matchedColorCount) / (slotSizeAsFlot * slotSizeAsFlot);
            }
        }
        return 0.0f;
    }

    __device__ __inline__ uint32_t getSolidDensity(float2 const& pos) const
    {
        auto index = toInt(pos.x) / _slotSize + toInt(pos.y) / _slotSize * _densityMapSize.x;
        if (index >= 0 && index < _densityMapSize.x * _densityMapSize.y) {
            return _solidDensityMap[index];
        }
        return 0;
    }

    __device__ __inline__ void addParticle(Energy* particle)
    {
        auto index = toInt(particle->pos.x) / _slotSize + toInt(particle->pos.y) / _slotSize * _densityMapSize.x;
        if (index >= 0 && index < _densityMapSize.x * _densityMapSize.y) {
            atomicAdd(&_energyParticleDensityMap[index], particle->energy);
        }
    }

    __device__ __inline__ void addFreeCell(Object* object)
    {
        auto index = toInt(object->pos.x) / _slotSize + toInt(object->pos.y) / _slotSize * _densityMapSize.x;
        if (index >= 0 && index < _densityMapSize.x * _densityMapSize.y) {
            auto color = calcMod(object->color, MAX_COLORS);
            if (color < 8) {
                alienAtomicAdd64(&_freeCellDensityMap1[index], static_cast<uint64_t>(1ull << (color * 8)));
                alienAtomicAdd64(&_freeCellDensityMap2[index], static_cast<uint64_t>(1ull << 16));
            } else {
                alienAtomicAdd64(&_freeCellDensityMap2[index], static_cast<uint64_t>((1ull << ((color - 8) * 8)) | (1ull << 16)));
            }
        }
    }

    __device__ __inline__ void addSolidObject(Object* object)
    {
        auto index = toInt(object->pos.x) / _slotSize + toInt(object->pos.y) / _slotSize * _densityMapSize.x;
        if (index >= 0 && index < _densityMapSize.x * _densityMapSize.y) {
            atomicAdd(&_solidDensityMap[index], 1u);
        }
    }

private:
    int _slotSize;
    int2 _densityMapSize;
    float* _energyParticleDensityMap;
    uint64_t* _freeCellDensityMap1;
    uint64_t* _freeCellDensityMap2;
    uint32_t* _solidDensityMap;
};
