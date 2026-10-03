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
        CudaMemoryManager::getInstance().acquireMemory<BarrierBounds>(1, _bounds);
        CHECK_FOR_DEVICE_ERRORS(cudaMemset(_bounds, 0, sizeof(BarrierBounds)));
    }

    __host__ __inline__ void free()
    {
        _solids.free();
        _barriers.free();
        CudaMemoryManager::getInstance().freeMemory(_bounds);
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
                updateBarrierBounds(object);
            }
        }
    }

    __device__ __inline__ void cleanup_system()
    {
        _solids.clear_system();
        _barriers.clear_system();
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            *_bounds = {0, 0};
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

    // A barrier crossed by a particle always has an endpoint within half of its connection length from the crossing point, plus the distance the
    // barrier moves. The barrier velocity was measured before the current forces accelerated the barriers.
    __device__ __inline__ float calcBarrierSearchRadius() const
    {
        auto maxBarrierVelocity = _bounds->maxVelocity + cudaSimulationParameters.maxAcceleration;
        return _bounds->maxConnectionLength / 2 + BarrierSearchMargin + maxBarrierVelocity * cudaSimulationParameters.timestepSize.value;
    }

    // Has to be called when a barrier is accelerated after the grid was built
    __device__ __inline__ void updateBarrierVelocity(Object* barrier)
    {
        auto velocity = Math::length(barrier->vel);
        if (velocity > _bounds->maxVelocity) {
            alienAtomicMax(&_bounds->maxVelocity, velocity);
        }
    }

private:
    static auto constexpr BarrierSearchMargin = 0.5f;

    // Upper bounds over all barriers, measured when the grid is built
    struct BarrierBounds
    {
        float maxConnectionLength;
        float maxVelocity;
    };

    __device__ __inline__ void updateBarrierBounds(Object* barrier)
    {
        auto maxConnectionLength = 0.0f;
        for (int i = 0; i < barrier->numConnections; ++i) {
            maxConnectionLength = max(maxConnectionLength, Math::length(_world.getCorrectedDirection(barrier->connections[i].object->pos - barrier->pos)));
        }
        if (maxConnectionLength > _bounds->maxConnectionLength) {
            alienAtomicMax(&_bounds->maxConnectionLength, maxConnectionLength);
        }
        updateBarrierVelocity(barrier);
    }

    WorldGeometry _world;
    OccupancyGrid _solids;
    OccupancyGrid _barriers;
    BarrierBounds* _bounds;
};
