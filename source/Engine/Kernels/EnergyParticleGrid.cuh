#pragma once

#include <vector>

#include "Entities.cuh"
#include "WorldGeometry.cuh"

// One energy particle at each integer position of the world
class EnergyParticleGrid
{
public:
    __host__ __inline__ void init(int2 const& size)
    {
        _world.init(size);
        _size = size;
        CudaMemoryManager::getInstance().acquireMemory<Energy*>(size.x * size.y, _map);
        _mapEntries.init();

        std::vector<Energy*> hostMap(size.x * size.y, 0);
        CHECK_FOR_DEVICE_ERRORS(cudaMemcpy(_map, hostMap.data(), sizeof(Energy*) * size.x * size.y, cudaMemcpyHostToDevice));
    }

    __host__ __inline__ void resize(int maxEntries) { _mapEntries.resize(maxEntries); }

    __device__ __inline__ void reset() { _mapEntries.reset(); }

    __host__ __inline__ void free()
    {
        CudaMemoryManager::getInstance().freeMemory(_map);
        _mapEntries.free();
    }

    __device__ __inline__ void set_block(int numEntities, Energy** entities)
    {
        if (0 == numEntities) {
            return;
        }

        __shared__ int* entrySubarray;
        if (0 == threadIdx.x) {
            entrySubarray = _mapEntries.getSubArray(numEntities);
        }
        __syncthreads();

        auto partition = calcThreadBlockPartition(numEntities);
        for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
            auto const& entity = entities[index];
            int2 posInt = {floorInt(entity->pos.x), floorInt(entity->pos.y)};
            _world.correctPosition(posInt);
            auto mapEntry = posInt.x + posInt.y * _size.x;
            _map[mapEntry] = entity;
            entrySubarray[index] = mapEntry;
        }
        __syncthreads();
    }

    __device__ __inline__ Energy* get(float2 const& pos) const
    {
        int2 posInt = {floorInt(pos.x), floorInt(pos.y)};
        _world.correctPosition(posInt);
        auto mapEntry = posInt.x + posInt.y * _size.x;
        return _map[mapEntry];
    }

    __device__ __inline__ void cleanup_system()
    {
        auto partition = calcSystemThreadPartition(_mapEntries.getNumEntries());
        for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
            auto const& mapEntry = _mapEntries.at(index);
            _map[mapEntry] = nullptr;
        }
    }

private:
    WorldGeometry _world;
    int2 _size;
    Energy** _map;
    Array<int> _mapEntries;
};
