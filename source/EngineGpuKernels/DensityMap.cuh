#pragma once

#include "Array.cuh"
#include "Base.cuh"
#include "CudaMemoryManager.cuh"
#include "Entity.cuh"

class DensityMap
{
public:
    __host__ __inline__ void init(int2 const& worldSize, int slotSize)
    {
        _densityMapSize = {worldSize.x / slotSize, worldSize.y / slotSize};
        CudaMemoryManager::getInstance().acquireMemory<float>(_densityMapSize.x * _densityMapSize.y, _energyParticleDensityMap);
        CudaMemoryManager::getInstance().acquireMemory<uint64_t>(_densityMapSize.x * _densityMapSize.y, _freeCellDensityMap);
        CudaMemoryManager::getInstance().acquireMemory<uint32_t>(_densityMapSize.x * _densityMapSize.y, _structureCellDensityMap);
        _slotSize = slotSize;
    }

    __host__ __inline__ void free()
    {
        CudaMemoryManager::getInstance().freeMemory(_energyParticleDensityMap);
        CudaMemoryManager::getInstance().freeMemory(_freeCellDensityMap);
        CudaMemoryManager::getInstance().freeMemory(_structureCellDensityMap);
    }

    __device__ __inline__ void clear()
    {
        auto const partition = calcSystemThreadPartition(_densityMapSize.x * _densityMapSize.y);
        for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
            _energyParticleDensityMap[index] = 0.0f;
            _freeCellDensityMap[index] = 0;
            _structureCellDensityMap[index] = 0;
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

    __device__ __inline__ float getFreeCellDensity(float2 const& pos, uint8_t restrictToColor) const
    {
        auto index = toInt(pos.x) / _slotSize + toInt(pos.y) / _slotSize * _densityMapSize.x;
        if (index >= 0 && index < _densityMapSize.x * _densityMapSize.y) {
            auto slotSizeAsFlot = toFloat(_slotSize);
            if (restrictToColor == 255) {
                // No color restriction - return total free cell count
                auto totalCount = (_freeCellDensityMap[index] >> 56) & 0xff;
                return toFloat(totalCount) / (slotSizeAsFlot * slotSizeAsFlot);
            } else {
                // Color restriction - return count for specific color
                auto colorCount = (_freeCellDensityMap[index] >> (restrictToColor * 8)) & 0xff;
                return toFloat(colorCount) / (slotSizeAsFlot * slotSizeAsFlot);
            }
        }
        return 0.0f;
    }

    __device__ __inline__ uint32_t getStructureDensity(float2 const& pos) const
    {
        auto index = toInt(pos.x) / _slotSize + toInt(pos.y) / _slotSize * _densityMapSize.x;
        if (index >= 0 && index < _densityMapSize.x * _densityMapSize.y) {
            return _structureCellDensityMap[index];
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
            // Increment both the color-specific count and the total count
            alienAtomicAdd64(&_freeCellDensityMap[index], static_cast<uint64_t>((1ull << (color * 8)) | (1ull << 56)));
        }
    }

    __device__ __inline__ void addStructureObject(Object* object)
    {
        auto index = toInt(object->pos.x) / _slotSize + toInt(object->pos.y) / _slotSize * _densityMapSize.x;
        if (index >= 0 && index < _densityMapSize.x * _densityMapSize.y) {
            atomicAdd(&_structureCellDensityMap[index], 1u);
        }
    }

private:
    int _slotSize;
    int2 _densityMapSize;
    float* _energyParticleDensityMap;
    uint64_t* _freeCellDensityMap;
    uint32_t* _structureCellDensityMap;
};
