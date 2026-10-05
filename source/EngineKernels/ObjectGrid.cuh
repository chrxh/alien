#pragma once

#include "cuda_runtime_api.h"

#include "Entities.cuh"
#include "SolidGrid.cuh"
#include "WorldGeometry.cuh"

// Lists of the objects at each integer position of the world
class ObjectGrid
{
public:
    __host__ __inline__ void init(int2 const& size)
    {
        _world.init(size);
        _size = size;
        CudaMemoryManager::getInstance().acquireMemory<int>(size.x * size.y, _mapHead);
        CHECK_FOR_DEVICE_ERRORS(cudaMemset(_mapHead, 0xff, sizeof(int) * size.x * size.y));  // 0xffffffff = -1 = empty
        _mapEntries.init();
        _records.init();
    }

    __host__ __inline__ void resize(int maxEntries)
    {
        _mapEntries.resize(maxEntries);
        _records.resize(maxEntries);
    }

    __device__ __inline__ void reset() { _mapEntries.reset(); }

    __host__ __inline__ void free()
    {
        CudaMemoryManager::getInstance().freeMemory(_mapHead);
        _mapEntries.free();
        _records.free();
    }

    __device__ __inline__ void set_block(int baseIndex, int numEntities, Object** objects)
    {
        if (0 == numEntities) {
            return;
        }

        __shared__ int* entrySubarray;
        if (0 == threadIdx.x) {
            entrySubarray = _mapEntries.getSubArray(numEntities);
        }
        __syncthreads();

        auto records = _records.getArray();
        auto partition = calcThreadBlockPartition(numEntities);
        for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
            auto object = objects[index];
            auto globalIndex = baseIndex + index;

            auto& record = records[globalIndex];
            record.initFrom(object);

            int2 posInt = {floorInt(object->pos.x), floorInt(object->pos.y)};
            _world.correctPosition(posInt);
            auto slot = posInt.x + posInt.y * _size.x;
            int slotIndex = atomicCAS(&_mapHead[slot], -1, globalIndex);
            for (int level = 0; level < 10; ++level) {
                if (slotIndex < 0) {
                    break;
                }
                slotIndex = atomicCAS(&records[slotIndex].nextObjectIndex, -1, globalIndex);
            }

            entrySubarray[index] = slot;
        }
        __syncthreads();
    }

    __device__ __inline__ int getFirstIndex(float2 const& pos) const
    {
        int2 posInt = {floorInt(pos.x), floorInt(pos.y)};
        _world.correctPosition(posInt);
        return _mapHead[posInt.x + posInt.y * _size.x];
    }

    __device__ __inline__ int getFirstIndex(int2 const& pos) const { return _mapHead[pos.x + pos.y * _size.x]; }

    __device__ __inline__ LightObject* getRecords() const { return _records.getArray(); }

    __device__ __inline__ void resetRecordLink(int index) { _records.at(index).nextObjectIndex = -1; }

    __device__ __inline__ Object* getFirst(float2 const& pos) const
    {
        auto index = getFirstIndex(pos);
        return index < 0 ? nullptr : _records.at(index).self;
    }

    __device__ __inline__ Object* getFirst(int2 const& pos) const
    {
        auto index = getFirstIndex(pos);
        return index < 0 ? nullptr : _records.at(index).self;
    }

    template <typename ExecFunc>
    __device__ __inline__ void executeForEach(float2 const& pos, float radius, int detached, ExecFunc const& execFunc) const
    {
        int2 posInt = {floorInt(pos.x), floorInt(pos.y)};
        int radiusInt = ceilf(radius);
        for (int dy = -radiusInt; dy <= radiusInt; ++dy) {
            for (int dx = -radiusInt; dx <= radiusInt; ++dx) {
                executeForEachInCell(int2{posInt.x + dx, posInt.y + dy}, pos, radius, detached, execFunc);
            }
        }
    }

    // Calls execFunc for the records of the solids at the positions within the radius.
    // The positions are distributed among numLanes threads, each calling this method with its own lane.
    template <typename ExecFunc>
    __device__ __inline__ void
    executeForEachSolidRecord(SolidGrid const& solidGrid, float2 const& pos, float radius, int lane, int numLanes, ExecFunc const& execFunc) const
    {
        int2 minPos{floorInt(pos.x - radius), floorInt(pos.y - radius)};
        int2 maxPos{floorInt(pos.x + radius), floorInt(pos.y + radius)};
        auto records = _records.getArray();
        int2 scanSize{maxPos.x - minPos.x + 1, maxPos.y - minPos.y + 1};
        for (int scanIndex = lane; scanIndex < scanSize.x * scanSize.y; scanIndex += numLanes) {
            int2 scanPos{minPos.x + scanIndex % scanSize.x, minPos.y + scanIndex / scanSize.x};
            auto deltaX = fmaxf(fmaxf(toFloat(scanPos.x) - pos.x, pos.x - toFloat(scanPos.x + 1)), 0.0f);
            auto deltaY = fmaxf(fmaxf(toFloat(scanPos.y) - pos.y, pos.y - toFloat(scanPos.y + 1)), 0.0f);
            if (deltaX * deltaX + deltaY * deltaY > radius * radius) {
                continue;
            }
            _world.correctPosition(scanPos);
            if (!solidGrid.hasSolid(scanPos)) {
                continue;
            }
            int index = _mapHead[scanPos.x + scanPos.y * _size.x];
            for (int level = 0; level < 10; ++level) {
                if (index < 0) {
                    break;
                }
                auto const& record = records[index];
                if (record.type == ObjectType_Solid) {
                    execFunc(record);
                }
                index = record.nextObjectIndex;
            }
        }
    }

    template <typename ExecFunc>
    __device__ __inline__ void executeForEach_block(float2 const& pos, float radius, int detached, ExecFunc const& execFunc) const
    {
        int2 posInt = {floorInt(pos.x), floorInt(pos.y)};
        int radiusInt = ceilf(radius);
        int scanLength = 2 * radiusInt + 1;
        for (int scanIndex = toInt(threadIdx.x); scanIndex < scanLength * scanLength; scanIndex += toInt(blockDim.x)) {
            int2 scanPos{posInt.x - radiusInt + scanIndex % scanLength, posInt.y - radiusInt + scanIndex / scanLength};
            executeForEachInCell(scanPos, pos, radius, detached, execFunc);
        }
    }

    template <typename ExecFunc>
    __device__ __inline__ void executeForEachInRing_block(float2 const& pos, float innerRadius, float outerRadius, int detached, ExecFunc const& execFunc) const
    {
        int2 posInt = {floorInt(pos.x), floorInt(pos.y)};
        int outerRadiusInt = ceilf(outerRadius) + 1;
        auto records = _records.getArray();
        for (int dy = -outerRadiusInt + toInt(threadIdx.x); dy <= outerRadiusInt; dy += toInt(blockDim.x)) {
            auto nearY = toFloat(max(abs(dy) - 1, 0));
            auto farY = toFloat(abs(dy) + 1);
            int outerDx = ceilf(sqrtf(max(outerRadius * outerRadius - nearY * nearY, 0.0f))) + 1;
            int innerDx = farY < innerRadius ? floorInt(sqrtf(innerRadius * innerRadius - farY * farY)) - 1 : -1;
            for (int dx = -outerDx; dx <= outerDx; ++dx) {
                if (abs(dx) <= innerDx) {
                    dx = innerDx;
                    continue;
                }
                int2 scanPos{posInt.x + dx, posInt.y + dy};
                _world.correctPosition(scanPos);
                int index = _mapHead[scanPos.x + scanPos.y * _size.x];
                for (int level = 0; level < 10; ++level) {
                    if (index < 0) {
                        break;
                    }
                    auto const& record = records[index];
                    auto slotObject = record.self;
                    auto delta = slotObject->pos - pos;
                    _world.correctDirection(delta);
                    auto distance = Math::length(delta);
                    if (distance > innerRadius && distance <= outerRadius && detached + slotObject->detached() != 1) {
                        execFunc(slotObject);
                    }
                    index = record.nextObjectIndex;
                }
            }
        }
    }

    __device__ __inline__ void cleanup_system()
    {
        auto partition = calcSystemThreadPartition(_mapEntries.getNumEntries());
        for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
            auto const& mapEntry = _mapEntries.at(index);
            _mapHead[mapEntry] = -1;
        }
    }

private:
    template <typename ExecFunc>
    __device__ __inline__ void executeForEachInCell(int2 scanPos, float2 const& pos, float radius, int detached, ExecFunc const& execFunc) const
    {
        _world.correctPosition(scanPos);
        auto records = _records.getArray();
        int index = _mapHead[scanPos.x + scanPos.y * _size.x];
        for (int level = 0; level < 10; ++level) {
            if (index < 0) {
                break;
            }
            auto const& record = records[index];
            auto slotObject = record.self;  // Read fields live: this runs after positions changed since the map was built
            if (Math::length(slotObject->pos - pos) <= radius && detached + slotObject->detached() != 1) {
                execFunc(slotObject);
            }
            index = record.nextObjectIndex;
        }
    }

    WorldGeometry _world;
    int2 _size;
    int* _mapHead;
    Array<int> _mapEntries;
    Array<LightObject> _records;
};
