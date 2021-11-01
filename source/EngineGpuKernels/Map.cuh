#pragma once

#include "Base.cuh"
#include "Cell.cuh"
#include "Particle.cuh"
#include "Math.cuh"
#include "cuda_runtime_api.h"

class MapInfo
{
public:
    __inline__ __host__ __device__ void init(int2 const& size) { _size = size; }

    __inline__ __host__ __device__ void mapPosCorrection(int2& pos) const
    {
        pos = {((pos.x % _size.x) + _size.x) % _size.x, ((pos.y % _size.y) + _size.y) % _size.y};
    }

    __inline__ __host__ __device__ void mapPosCorrection(float2& pos) const
    {
        int2 intPart{floorInt(pos.x), floorInt(pos.y)};
        float2 fracPart = {pos.x - intPart.x, pos.y - intPart.y};
        mapPosCorrection(intPart);
        pos = {static_cast<float>(intPart.x) + fracPart.x, static_cast<float>(intPart.y) + fracPart.y};
    }

    __inline__ __device__ void mapDisplacementCorrection(float2& disp) const
    {
        disp.x = remainderf(disp.x, _size.x);
        disp.y = remainderf(disp.y, _size.y);
    }

    __inline__ __device__ float mapDistance(float2 const& p, float2 const& q) const
    {
        float2 d = {p.x - q.x, p.y - q.y};
        mapDisplacementCorrection(d);
        return sqrt(d.x * d.x + d.y * d.y);
    }

    __inline__ __device__ float2 correctionIncrement(float2 pos1, float2 pos2) const
    {
        float2 result{0.0f, 0.0f};
        if (pos2.x - pos1.x > _size.x / 2) {
            result.x = -_size.x;
        }
        if (pos1.x - pos2.x > _size.x / 2) {
            result.x = _size.x;
        }
        if (pos2.y - pos1.y > _size.y / 2) {
            result.y = -_size.y;
        }
        if (pos1.y - pos2.y > _size.y / 2) {
            result.y = _size.y;
        }
        return result;
    }

    __inline__ __device__ int getMaxRadius() const { return min(_size.x, _size.y) / 4; }

protected:
    int2 _size;
};

template <typename T>
class BasicMap : public MapInfo
{
public:
};

class CellMap : public MapInfo
{
public:
    __host__ __inline__ void init(int2 const& size)
    {
        MapInfo::init(size);
        CudaMemoryManager::getInstance().acquireMemory<Cell*>(size.x * size.y * 2, _map);
        _mapEntries.init();

        std::vector<Cell*> hostMap(size.x * size.y * 2, 0);
        checkCudaErrors(cudaMemcpy(_map, hostMap.data(), sizeof(Cell*) * size.x * size.y * 2, cudaMemcpyHostToDevice));
    }

    __host__ __inline__ void resize(int maxEntries) { _mapEntries.resize(maxEntries); }

    __device__ __inline__ void reset() { _mapEntries.reset(); }

    __host__ __inline__ void free()
    {
        CudaMemoryManager::getInstance().freeMemory(_map);
        _mapEntries.free();
    }

    __device__ __inline__ void set_block(int numEntities, Cell** entities)
    {
        if (0 == numEntities) {
            return;
        }

        __shared__ int* entrySubarray;
        if (0 == threadIdx.x) {
            entrySubarray = _mapEntries.getNewSubarray(numEntities);
        }
        __syncthreads();

        auto partition = calcPartition(numEntities, threadIdx.x, blockDim.x);
        for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
            auto const& entity = entities[index];
            int2 posInt = {floorInt(entity->absPos.x), floorInt(entity->absPos.y)};
            mapPosCorrection(posInt);
            auto mapEntry = (posInt.x + posInt.y * _size.x) * 2;
            auto old = reinterpret_cast<Cell*>(atomicCAS(
                reinterpret_cast<unsigned long long int*>(&_map[mapEntry]),
                reinterpret_cast<unsigned long long int>(nullptr),
                reinterpret_cast<unsigned long long int>(entity)));
            if (old != nullptr) {
                atomicExch(&_map[mapEntry + 1], entity);
            }
            entrySubarray[index] = mapEntry;
        }
        __syncthreads();
    }

    __device__ __inline__ void get(Cell* cells[], int& numCells, float2 const& pos) const
    {
        int2 posInt = {floorInt(pos.x), floorInt(pos.y)};
        numCells = 0;
        for (int dx = -1; dx <= 1; ++dx) {
            for (int dy = -1; dy <= 1; ++dy) {
                int2 scanPos{posInt.x + dx, posInt.y + dy};
                mapPosCorrection(scanPos);

                auto mapEntry = (scanPos.x + scanPos.y * _size.x) * 2;
                if (cells[numCells] = _map[mapEntry]) {
                    ++numCells;
                    if (cells[numCells] = _map[mapEntry + 1]) {
                        ++numCells;
                    }
                }
            }
        }
    }

    __device__ __inline__ void get(Cell* cells[], int arraySize, int& numCells, float2 const& pos, float radius) const
    {
        int2 posInt = {floorInt(pos.x), floorInt(pos.y)};
        numCells = 0;
        int radiusInt = ceilf(radius);
        for (int dx = -radiusInt; dx <= radiusInt; ++dx) {
            for (int dy = -radiusInt; dy <= radiusInt; ++dy) {
                int2 scanPos{posInt.x + dx, posInt.y + dy};
                mapPosCorrection(scanPos);

                auto mapEntry = (scanPos.x + scanPos.y * _size.x) * 2;
                auto cell1 = _map[mapEntry];
                if (cell1 && Math::length(cell1->absPos - pos) <= radius && numCells < arraySize) {
                    cells[numCells] = cell1;
                    ++numCells;

                    auto cell2 = _map[mapEntry + 1];
                    if (cell2 && Math::length(cell2->absPos - pos) <= radius && numCells < arraySize) {
                        cells[numCells] = cell2;
                        ++numCells;
                    }
                }
            }
        }
    }

    __device__ __inline__ Cell* getFirst(float2 const& pos) const
    {
        int2 posInt = {floorInt(pos.x), floorInt(pos.y)};
        mapPosCorrection(posInt);
        auto mapEntry = (posInt.x + posInt.y * _size.x) * 2;
        return _map[mapEntry];
    }

    __device__ __inline__ void cleanup_system()
    {
        auto partition =
            calcPartition(_mapEntries.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);
        for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
            auto const& mapEntry = _mapEntries.at(index);
            _map[mapEntry] = nullptr;
            _map[mapEntry + 1] = nullptr;
        }
    }

private:
    Cell** _map;
    Array<int> _mapEntries;
};

class ParticleMap : public MapInfo
{
public:
    __host__ __inline__ void init(int2 const& size)
    {
        MapInfo::init(size);
        CudaMemoryManager::getInstance().acquireMemory<Particle*>(size.x * size.y, _map);
        _mapEntries.init();

        std::vector<Particle*> hostMap(size.x * size.y, 0);
        checkCudaErrors(cudaMemcpy(_map, hostMap.data(), sizeof(Particle*) * size.x * size.y, cudaMemcpyHostToDevice));
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
            entrySubarray = _mapEntries.getNewSubarray(numEntities);
        }
        __syncthreads();

        auto partition = calcPartition(numEntities, threadIdx.x, blockDim.x);
        for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
            auto const& entity = entities[index];
            int2 posInt = {floorInt(entity->absPos.x), floorInt(entity->absPos.y)};
            mapPosCorrection(posInt);
            auto mapEntry = posInt.x + posInt.y * _size.x;
            _map[mapEntry] = entity;
            entrySubarray[index] = mapEntry;
        }
        __syncthreads();
    }

    __device__ __inline__ Particle* get(float2 const& pos) const
    {
        int2 posInt = { floorInt(pos.x), floorInt(pos.y) };
        mapPosCorrection(posInt);
        auto mapEntry = posInt.x + posInt.y * _size.x;
        return _map[mapEntry];
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
    Particle** _map;
    Array<int> _mapEntries;
};
