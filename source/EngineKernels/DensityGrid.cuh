#pragma once

#include "Entities.cuh"

// Densities of energy particles and free cells in slots of SlotSize x SlotSize positions
class DensityGrid
{
public:
    static int constexpr SlotSize = 8;

    __host__ __inline__ void init(int2 const& worldSize)
    {
        _numSlots = {worldSize.x / SlotSize, worldSize.y / SlotSize};
        CudaMemoryManager::getInstance().acquireMemory<float>(_numSlots.x * _numSlots.y, _energyParticleDensities);
        CudaMemoryManager::getInstance().acquireMemory<uint64_t>(_numSlots.x * _numSlots.y, _freeCellDensities1);
        CudaMemoryManager::getInstance().acquireMemory<uint64_t>(_numSlots.x * _numSlots.y, _freeCellDensities2);
    }

    __host__ __inline__ void free()
    {
        CudaMemoryManager::getInstance().freeMemory(_energyParticleDensities);
        CudaMemoryManager::getInstance().freeMemory(_freeCellDensities1);
        CudaMemoryManager::getInstance().freeMemory(_freeCellDensities2);
    }

    __device__ __inline__ void clear()
    {
        auto const partition = calcSystemThreadPartition(_numSlots.x * _numSlots.y);
        for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
            _energyParticleDensities[index] = 0.0f;
            _freeCellDensities1[index] = 0;
            _freeCellDensities2[index] = 0;
        }
    }

    __device__ __inline__ float getEnergyParticleDensity(float2 const& pos) const
    {
        auto index = toInt(pos.x) / SlotSize + toInt(pos.y) / SlotSize * _numSlots.x;
        if (index >= 0 && index < _numSlots.x * _numSlots.y) {
            auto slotSizeAsFlot = toFloat(SlotSize);
            return _energyParticleDensities[index] / (slotSizeAsFlot * slotSizeAsFlot);
        }
        return 0.0f;
    }

    __device__ __inline__ float getFreeCellDensity(float2 const& pos, uint16_t restrictToColors) const
    {
        auto index = toInt(pos.x) / SlotSize + toInt(pos.y) / SlotSize * _numSlots.x;
        if (index >= 0 && index < _numSlots.x * _numSlots.y) {
            auto slotSizeAsFlot = toFloat(SlotSize);
            if (restrictToColors == 0x3ff) {
                auto totalCount = (_freeCellDensities2[index] >> 16) & 0xff;
                return toFloat(totalCount) / (slotSizeAsFlot * slotSizeAsFlot);
            } else {
                int matchedColorCount = 0;
                for (int color = 0; color < MAX_COLORS; ++color) {
                    if ((restrictToColors >> color) & 1) {
                        if (color < 8) {
                            matchedColorCount += (_freeCellDensities1[index] >> (color * 8)) & 0xff;
                        } else {
                            matchedColorCount += (_freeCellDensities2[index] >> ((color - 8) * 8)) & 0xff;
                        }
                    }
                }
                return toFloat(matchedColorCount) / (slotSizeAsFlot * slotSizeAsFlot);
            }
        }
        return 0.0f;
    }

    __device__ __inline__ void addParticle(Energy* particle)
    {
        auto index = toInt(particle->pos.x) / SlotSize + toInt(particle->pos.y) / SlotSize * _numSlots.x;
        if (index >= 0 && index < _numSlots.x * _numSlots.y) {
            atomicAdd(&_energyParticleDensities[index], particle->energy);
        }
    }

    __device__ __inline__ void addFreeCell(Object* object)
    {
        auto index = toInt(object->pos.x) / SlotSize + toInt(object->pos.y) / SlotSize * _numSlots.x;
        if (index >= 0 && index < _numSlots.x * _numSlots.y) {
            auto color = calcMod(object->color, MAX_COLORS);
            if (color < 8) {
                alienAtomicAdd64(&_freeCellDensities1[index], static_cast<uint64_t>(1ull << (color * 8)));
                alienAtomicAdd64(&_freeCellDensities2[index], static_cast<uint64_t>(1ull << 16));
            } else {
                alienAtomicAdd64(&_freeCellDensities2[index], static_cast<uint64_t>((1ull << ((color - 8) * 8)) | (1ull << 16)));
            }
        }
    }

private:
    int2 _numSlots;
    float* _energyParticleDensities;
    uint64_t* _freeCellDensities1;
    uint64_t* _freeCellDensities2;
};
