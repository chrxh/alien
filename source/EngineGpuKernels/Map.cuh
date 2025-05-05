#pragma once

#include "cuda_runtime_api.h"

#include "Base.cuh"
#include "Object.cuh"
#include "Math.cuh"
#include "Array.cuh"

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

class CellMap : public BaseMap
{
public:
    __host__ __inline__ void init(int2 const& size)
    {
        BaseMap::init(size);
        CudaMemoryManager::getInstance().acquireMemory<Cell*>(size.x * size.y, _map);
        _mapEntries.init();

        std::vector<Cell*> hostMap(size.x * size.y, 0);
        CHECK_FOR_CUDA_ERROR(cudaMemcpy(_map, hostMap.data(), sizeof(Cell*) * size.x * size.y, cudaMemcpyHostToDevice));
    }

    __host__ __inline__ void resize(int maxEntries) { _mapEntries.resize(maxEntries); }


    __device__ __inline__ void reset() { _mapEntries.reset(); }

    __host__ __inline__ void free()
    {
        CudaMemoryManager::getInstance().freeMemory(_map);
        _mapEntries.free();
    }

    __device__ __inline__ void set_block(int numEntities, Cell** cells)
    {
        if (0 == numEntities) {
            return;
        }

        __shared__ int* entrySubarray;
        if (0 == threadIdx.x) {
            entrySubarray = _mapEntries.getSubArray(numEntities);
        }
        __syncthreads();

        auto partition = calcPartition(numEntities, threadIdx.x, blockDim.x);
        for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
            auto const& cell = cells[index];
            int2 posInt = {floorInt(cell->pos.x), floorInt(cell->pos.y)};
            correctPosition(posInt);
            auto slot = posInt.x + posInt.y * _size.x;
            Cell* slotCell = reinterpret_cast<Cell*>(atomicCAS(
                reinterpret_cast<unsigned long long int*>(&_map[slot]),
                reinterpret_cast<unsigned long long int>(nullptr),
                reinterpret_cast<unsigned long long int>(cell)));
            for (int level = 0; level < 10; ++level) {
                if (!slotCell) {
                    break;
                }
                slotCell = reinterpret_cast<Cell*>(atomicCAS(
                    reinterpret_cast<unsigned long long int*>(&slotCell->nextCell),
                    reinterpret_cast<unsigned long long int>(nullptr),
                    reinterpret_cast<unsigned long long int>(cell)));
            }

            entrySubarray[index] = slot;
        }
        __syncthreads();
    }

    __device__ __inline__ Cell* getFirst(float2 const& pos) const
    {
        int2 posInt = {floorInt(pos.x), floorInt(pos.y)};
        correctPosition(posInt);
        return _map[posInt.x + posInt.y * _size.x];
    }

    __device__ __inline__ Cell* getFirst(int2 const& pos) const
    {
        return _map[pos.x + pos.y * _size.x];
    }

    template <typename MatchFunc>
    __device__ __inline__ void getMatchingCells(Cell* cells[], int arraySize, int& numCells, float2 const& pos, float radius, int detached, MatchFunc matchFunc)
        const
    {
        int2 posInt = {floorInt(pos.x), floorInt(pos.y)};
        numCells = 0;
        int radiusInt = ceilf(radius);
        for (int dx = -radiusInt; dx <= radiusInt; ++dx) {
            for (int dy = -radiusInt; dy <= radiusInt; ++dy) {
                int2 scanPos{posInt.x + dx, posInt.y + dy};
                correctPosition(scanPos);
                int slot = scanPos.x + scanPos.y * _size.x;
                auto slotCell = _map[slot];
                for (int level = 0; level < 10; ++level) {
                    if (numCells == arraySize) {
                        return;
                    }
                    if (!slotCell) {
                        break;
                    }
                    if (Math::length(slotCell->pos - pos) <= radius && detached + slotCell->detached != 1 && matchFunc(slotCell)) {
                        cells[numCells] = slotCell;
                        ++numCells;
                    }
                    slotCell = slotCell->nextCell;
                }
            }
        }
    }

    template <typename ExecFunc>
    __device__ __inline__ void executeForEach(float2 const& pos, float radius, int detached, ExecFunc const& execFunc) const
    {
        int2 posInt = {floorInt(pos.x), floorInt(pos.y)};
        int radiusInt = ceilf(radius);
        for (int dy = -radiusInt; dy <= radiusInt; ++dy) {
            for (int dx = -radiusInt; dx <= radiusInt; ++dx) {
                int2 scanPos{posInt.x + dx, posInt.y + dy};
                correctPosition(scanPos);
                int slot = scanPos.x + scanPos.y * _size.x;
                auto slotCell = _map[slot];
                for (int level = 0; level < 10; ++level) {
                    if (!slotCell) {
                        break;
                    }
                    if (Math::length(slotCell->pos - pos) <= radius && detached + slotCell->detached != 1) {
                        execFunc(slotCell);
                    }
                    slotCell = slotCell->nextCell;
                }
            }
        }
    }

    __device__ __inline__ void cleanup_system()
    {
        auto partition =
            calcPartition(_mapEntries.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);
        for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
            auto const& mapEntry = _mapEntries.at(index);
            _map[mapEntry] = nullptr;
        }
    }

private:
    Cell** _map;
    Array<int> _mapEntries;
};

class ParticleMap : public BaseMap
{
public:
    __host__ __inline__ void init(int2 const& size)
    {
        BaseMap::init(size);
        CudaMemoryManager::getInstance().acquireMemory<Particle*>(size.x * size.y, _map);
        _mapEntries.init();

        std::vector<Particle*> hostMap(size.x * size.y, 0);
        CHECK_FOR_CUDA_ERROR(cudaMemcpy(_map, hostMap.data(), sizeof(Particle*) * size.x * size.y, cudaMemcpyHostToDevice));
    }

    __host__ __inline__ void resize(int maxEntries) { _mapEntries.resize(maxEntries); }

    __device__ __inline__ void reset() { _mapEntries.reset(); }

    __host__ __inline__ void free()
    {
        CudaMemoryManager::getInstance().freeMemory(_map);
        _mapEntries.free();
    }

    __device__ __inline__ void set_block(int numEntities, Particle** entities)
    {
        if (0 == numEntities) {
            return;
        }

        __shared__ int* entrySubarray;
        if (0 == threadIdx.x) {
            entrySubarray = _mapEntries.getSubArray(numEntities);
        }
        __syncthreads();

        auto partition = calcPartition(numEntities, threadIdx.x, blockDim.x);
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

    __device__ __inline__ Particle* get(float2 const& pos) const
    {
        int2 posInt = {floorInt(pos.x), floorInt(pos.y)};
        correctPosition(posInt);
        auto mapEntry = posInt.x + posInt.y * _size.x;
        return _map[mapEntry];
    }

    __device__ __inline__ void cleanup_system()
    {
        auto partition = calcPartition(_mapEntries.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);
        for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
            auto const& mapEntry = _mapEntries.at(index);
            _map[mapEntry] = nullptr;
        }
    }

private:
    Particle** _map;
    Array<int> _mapEntries;
};
