#pragma once

#include "ConstantMemory.cuh"
#include "Entities.cuh"
#include "OccupancyGrid.cuh"
#include "WorldGeometry.cuh"

// Barriers are solid and static objects, energy particles bounce off the connections between them.
// Marks the positions of solids (for sensors) and of barriers (for energy particles).
class BarrierGrid
{
public:
    __host__ __inline__ void init(int2 const& worldSize)
    {
        _world.init(worldSize);
        _solids.init(worldSize, OccupancyGrid::Levels::PositionsAndBlocks);
        _barriers.init(worldSize, OccupancyGrid::Levels::Positions);
        CudaMemoryManager::getInstance().acquireMemory<float>(1, _maxReach);
        CHECK_FOR_DEVICE_ERRORS(cudaMemset(_maxReach, 0, sizeof(float)));
    }

    __host__ __inline__ void free()
    {
        _solids.free();
        _barriers.free();
        CudaMemoryManager::getInstance().freeMemory(_maxReach);
    }

    __device__ __inline__ void set_block(int numEntities, Object** objects)
    {
        auto partition = calcThreadBlockPartition(numEntities);
        for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
            auto object = objects[index];
            int2 posInt = {floorInt(object->pos.x), floorInt(object->pos.y)};
            _world.correctPosition(posInt);
            if (object->type == ObjectType_Solid) {
                _solids.set(posInt);
            }
            if (object->type == ObjectType_Solid || object->isStatic()) {
                _barriers.set(posInt);
                updateMaxReach(object);
            }
        }
    }

    __device__ __inline__ void cleanup_system()
    {
        _solids.clear_system();
        _barriers.clear_system();
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            *_maxReach = 0;
        }
    }

    __device__ __inline__ bool hasSolid(int2 const& pos) const { return _solids.isSet(pos); }

    __device__ __inline__ bool hasSolid(int2 const& minPos, int2 const& maxPos) const { return _solids.isAnySet(minPos, maxPos); }

    // False if no solid lies in the block of the position or within one unit around it, the blocks are squares of size OccupancyGrid::BlockSize.
    // The position must be corrected.
    __device__ __inline__ bool hasSolidNearBlockOf(float2 const& pos) const { return _solids.isAnySetNearBlockOf({floorInt(pos.x), floorInt(pos.y)}); }

    // The position must be corrected
    __device__ __inline__ bool hasBarrier(int2 const& pos) const { return _barriers.isSet(pos); }

    // Checks the positions in the bounding square of the circle
    __device__ __inline__ bool hasBarrier(float2 const& pos, float radius) const
    {
        return _barriers.isAnySet({floorInt(pos.x - radius), floorInt(pos.y - radius)}, {floorInt(pos.x + radius), floorInt(pos.y + radius)});
    }

    // A barrier crossed by a particle always has an endpoint within its reach from the crossing point. The reach was measured before the
    // barriers were accelerated in the current timestep. Accelerations by force fields are small enough to be covered by the margin.
    __device__ __inline__ float calcBarrierSearchRadius() const
    {
        return *_maxReach + BarrierSearchMargin + cudaSimulationParameters.maxAcceleration * cudaSimulationParameters.timestepSize.value;
    }

private:
    static auto constexpr BarrierSearchMargin = 0.5f;

    __device__ __inline__ void updateMaxReach(Object* barrier)
    {
        auto reach = calcReach(barrier);
        if (reach > *_maxReach) {
            alienAtomicMax(_maxReach, reach);
        }
    }

    // Half of the longest connection plus the distance the barrier moves in a timestep
    __device__ __inline__ float calcReach(Object* barrier) const
    {
        auto maxConnectionLength = 0.0f;
        for (int i = 0; i < barrier->numConnections; ++i) {
            maxConnectionLength = max(maxConnectionLength, Math::length(_world.getCorrectedDirection(barrier->connections[i].object->pos - barrier->pos)));
        }
        return maxConnectionLength / 2 + Math::length(barrier->vel) * cudaSimulationParameters.timestepSize.value;
    }

    WorldGeometry _world;
    OccupancyGrid _solids;
    OccupancyGrid _barriers;
    float* _maxReach;
};
