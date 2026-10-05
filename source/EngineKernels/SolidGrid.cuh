#pragma once

#include "ConstantMemory.cuh"
#include "Entities.cuh"
#include "OccupancyGrid.cuh"
#include "WorldGeometry.cuh"

// Marks the positions of the solids. Sensors detect solids with it and energy particles bounce off the connections between solids.
class SolidGrid
{
public:
    __host__ __inline__ void init(int2 const& worldSize)
    {
        _world.init(worldSize);
        _solids.init(worldSize);
        CudaMemoryManager::getInstance().acquireMemory<float>(1, _maxReach);
        CHECK_FOR_DEVICE_ERRORS(cudaMemset(_maxReach, 0, sizeof(float)));
    }

    __host__ __inline__ void free()
    {
        _solids.free();
        CudaMemoryManager::getInstance().freeMemory(_maxReach);
    }

    __device__ __inline__ void set_block(int numEntities, Object** objects)
    {
        auto partition = calcThreadBlockPartition(numEntities);
        for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
            auto object = objects[index];
            if (object->type != ObjectType_Solid) {
                continue;
            }
            int2 posInt = {floorInt(object->pos.x), floorInt(object->pos.y)};
            _world.correctPosition(posInt);
            _solids.set(posInt);
            updateMaxReach(object);
        }
    }

    __device__ __inline__ void cleanup_system()
    {
        _solids.clear_system();
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            *_maxReach = 0;
        }
    }

    // The position must be corrected
    __device__ __inline__ bool hasSolid(int2 const& pos) const { return _solids.isSet(pos); }

    __device__ __inline__ bool hasSolid(int2 const& minPos, int2 const& maxPos) const { return _solids.isAnySet(minPos, maxPos); }

    // Checks the positions in the bounding square of the circle
    __device__ __inline__ bool hasSolid(float2 const& pos, float radius) const
    {
        return _solids.isAnySet({floorInt(pos.x - radius), floorInt(pos.y - radius)}, {floorInt(pos.x + radius), floorInt(pos.y + radius)});
    }

    // False if no solid lies in the block of the position or within one unit around it, the blocks are squares of size OccupancyGrid::BlockSize.
    // The position must be corrected.
    __device__ __inline__ bool hasSolidNearBlockOf(float2 const& pos) const { return _solids.isAnySetNearBlockOf({floorInt(pos.x), floorInt(pos.y)}); }

    // A connection between solids crossed by a particle always has an endpoint within its reach from the crossing point. The reach was measured
    // before the solids were accelerated in the current timestep. Accelerations by force fields are small enough to be covered by the margin.
    __device__ __inline__ float calcSolidSearchRadius() const
    {
        return *_maxReach + SolidSearchMargin + cudaSimulationParameters.maxAcceleration * cudaSimulationParameters.timestepSize.value;
    }

private:
    static auto constexpr SolidSearchMargin = 0.5f;

    __device__ __inline__ void updateMaxReach(Object* solid)
    {
        auto reach = calcReach(solid);
        if (reach > *_maxReach) {
            alienAtomicMax(_maxReach, reach);
        }
    }

    // Half of the longest connection plus the distance the solid moves in a timestep
    __device__ __inline__ float calcReach(Object* solid) const
    {
        auto maxConnectionLength = 0.0f;
        for (int i = 0; i < solid->numConnections; ++i) {
            maxConnectionLength = max(maxConnectionLength, Math::length(_world.getCorrectedDirection(solid->connections[i].object->pos - solid->pos)));
        }
        return maxConnectionLength / 2 + Math::length(solid->vel) * cudaSimulationParameters.timestepSize.value;
    }

    WorldGeometry _world;
    OccupancyGrid _solids;
    float* _maxReach;
};
