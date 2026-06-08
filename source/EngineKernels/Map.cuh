#pragma once

#include "cuda_runtime_api.h"

#include "Entities.cuh"

class BaseMap
{
public:
    __inline__ __host__ __device__ void init(int2 const& size) { _size = size; }

    __inline__ __host__ __device__ void correctPosition(int2& pos) const
    {
        pos = {((pos.x % _size.x) + _size.x) % _size.x, ((pos.y % _size.y) + _size.y) % _size.y};
    }

    __inline__ __host__ __device__ void correctPosition(float2& pos) const
    {
        int2 intPart{floorInt(pos.x), floorInt(pos.y)};
        float2 fracPart = {pos.x - toFloat(intPart.x), pos.y - toFloat(intPart.y)};
        correctPosition(intPart);
        pos = {static_cast<float>(intPart.x) + fracPart.x, static_cast<float>(intPart.y) + fracPart.y};
    }

    __inline__ __device__ float2 getCorrectedPosition(float2 const& pos) const
    {
        auto copy = pos;
        correctPosition(copy);
        return copy;
    }

    __inline__ __device__ void correctDirection(float2& disp) const
    {
        disp.x = remainderf(disp.x, toFloat(_size.x));
        disp.y = remainderf(disp.y, toFloat(_size.y));
    }

    __inline__ __device__ float2 getCorrectedDirection(float2 const& disp) const
    {
        return {remainderf(disp.x, toFloat(_size.x)), remainderf(disp.y, toFloat(_size.y))};
    }

    __inline__ __device__ float getDistance(float2 const& p, float2 const& q) const
    {
        float2 d = {p.x - q.x, p.y - q.y};
        correctDirection(d);
        return sqrt(d.x * d.x + d.y * d.y);
    }

    __inline__ __device__ float2 getCorrectionIncrement(float2 pos1, float2 pos2) const
    {
        auto delta = pos1 - pos2 + toFloat2(_size) / 2;
        return {delta.x - Math::modulo(delta.x, toFloat(_size.x)), delta.y - Math::modulo(delta.y, toFloat(_size.y))};
    }

    __inline__ __device__ int getMaxRadius() const { return min(_size.x, _size.y) / 4; }

protected:
    int2 _size;
};

template <typename T>
class BasicMap : public BaseMap
{
public:
};

// Compact mirror of the Object fields read during the spatial neighbor scan. Built each timestep in set_block and
// indexed by global object index; the cell map and per-cell chain use indices into this array, so a scan reads ~40
// contiguous bytes per neighbor instead of dereferencing the full ~500-byte Object.
struct __align__(8) ScanRecord
{
    float2 pos;
    float2 vel;
    float density;
    int nextObjectIndex;  // Next object in the same cell, -1 = end of chain
    Object* self;         // Full object, used for rare branches and force write-back
    ObjectType type;
    uint8_t numConnections;
    uint8_t flags;  // bit0 = fixed, bit1 = detached, bit2 = sticky

    __device__ __inline__ bool isFixed() const { return flags & 1; }
    __device__ __inline__ int detached() const { return (flags >> 1) & 1; }
    __device__ __inline__ bool isSticky() const { return flags & 4; }

    __device__ __inline__ float getMassForSPH() const
    {
        if (type == ObjectType_Fluid) {
            return 0.1f;
        } else if (type == ObjectType_Solid) {
            return 2.0f;
        } else {
            return 1.0f;
        }
    }
};

class ObjectMap : public BaseMap
{
public:
    __host__ __inline__ void init(int2 const& size)
    {
        BaseMap::init(size);
        CudaMemoryManager::getInstance().acquireMemory<int>(size.x * size.y, _mapHead);
        CHECK_FOR_DEVICE_ERRORS(cudaMemset(_mapHead, 0xFF, sizeof(int) * size.x * size.y));  // 0xFFFFFFFF = -1 = empty
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
            record.pos = object->pos;
            record.vel = object->vel;
            record.density = object->density;
            record.self = object;
            record.type = object->type;
            record.numConnections = object->numConnections;
            record.flags = (object->fixed ? 1 : 0) | (object->detached ? 2 : 0) | (object->sticky ? 4 : 0);
            // nextObjectIndex is reset to -1 in ObjectProcessor::init and only changed by atomicCAS below

            int2 posInt = {floorInt(object->pos.x), floorInt(object->pos.y)};
            correctPosition(posInt);
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
        correctPosition(posInt);
        return _mapHead[posInt.x + posInt.y * _size.x];
    }

    __device__ __inline__ int getFirstIndex(int2 const& pos) const { return _mapHead[pos.x + pos.y * _size.x]; }

    __device__ __inline__ ScanRecord* getRecords() const { return _records.getArray(); }

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

    template <typename MatchFunc>
    __device__ __inline__ void
    getMatchingObjects(Object* objects[], int arraySize, int& numObjects, float2 const& pos, float radius, int detached, MatchFunc matchFunc) const
    {
        int2 posInt = {floorInt(pos.x), floorInt(pos.y)};
        numObjects = 0;
        int radiusInt = ceilf(radius);
        auto records = _records.getArray();
        for (int dx = -radiusInt; dx <= radiusInt; ++dx) {
            for (int dy = -radiusInt; dy <= radiusInt; ++dy) {
                int2 scanPos{posInt.x + dx, posInt.y + dy};
                correctPosition(scanPos);
                int index = _mapHead[scanPos.x + scanPos.y * _size.x];
                for (int level = 0; level < 10; ++level) {
                    if (numObjects == arraySize) {
                        return;
                    }
                    if (index < 0) {
                        break;
                    }
                    auto const& record = records[index];
                    auto slotObject = record.self;  // Read fields live: this runs after positions changed since the map was built
                    if (Math::length(slotObject->pos - pos) <= radius && detached + slotObject->detached != 1 && matchFunc(slotObject)) {
                        objects[numObjects] = slotObject;
                        ++numObjects;
                    }
                    index = record.nextObjectIndex;
                }
            }
        }
    }

    template <typename ExecFunc>
    __device__ __inline__ void executeForEach(float2 const& pos, float radius, int detached, ExecFunc const& execFunc) const
    {
        int2 posInt = {floorInt(pos.x), floorInt(pos.y)};
        int radiusInt = ceilf(radius);
        auto records = _records.getArray();
        for (int dy = -radiusInt; dy <= radiusInt; ++dy) {
            for (int dx = -radiusInt; dx <= radiusInt; ++dx) {
                int2 scanPos{posInt.x + dx, posInt.y + dy};
                correctPosition(scanPos);
                int index = _mapHead[scanPos.x + scanPos.y * _size.x];
                for (int level = 0; level < 10; ++level) {
                    if (index < 0) {
                        break;
                    }
                    auto const& record = records[index];
                    auto slotObject = record.self;  // Read fields live: this runs after positions changed since the map was built
                    if (Math::length(slotObject->pos - pos) <= radius && detached + slotObject->detached != 1) {
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
    int* _mapHead;
    Array<int> _mapEntries;
    Array<ScanRecord> _records;
};

class EnergyMap : public BaseMap
{
public:
    __host__ __inline__ void init(int2 const& size)
    {
        BaseMap::init(size);
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
            correctPosition(posInt);
            auto mapEntry = posInt.x + posInt.y * _size.x;
            _map[mapEntry] = entity;
            entrySubarray[index] = mapEntry;
        }
        __syncthreads();
    }

    __device__ __inline__ Energy* get(float2 const& pos) const
    {
        int2 posInt = {floorInt(pos.x), floorInt(pos.y)};
        correctPosition(posInt);
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
    Energy** _map;
    Array<int> _mapEntries;
};
