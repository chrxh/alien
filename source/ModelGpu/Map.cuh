#pragma once

#include "device_functions.h"

class BasicMap
{
public:
	__inline__ __host__ __device__ void init(int2 const& size)
	{
		_size = size;
	}

	__inline__ __host__ __device__ void mapPosCorrection(int2 &pos) const
	{
		pos = { ((pos.x % _size.x) + _size.x) % _size.x, ((pos.y % _size.y) + _size.y) % _size.y };
	}

	__inline__ __host__ __device__ void mapPosCorrection(float2 &pos) const
	{
		int2 intPart{ floorInt(pos.x), floorInt(pos.y) };
		float2 fracPart = { pos.x - intPart.x, pos.y - intPart.y };
		mapPosCorrection(intPart);
		pos = { static_cast<float>(intPart.x) + fracPart.x, static_cast<float>(intPart.y) + fracPart.y };
	}

	__inline__ __device__ void mapDisplacementCorrection(float2 &disp) const
	{
		disp.x = remainderf(disp.x, _size.x / 2);
		disp.y = remainderf(disp.y, _size.y / 2);
	}

	__inline__ __device__ float mapDistance(float2 const &p, float2 const &q) const
	{
		float2 d = { p.x - q.x, p.y - q.y };
		mapDisplacementCorrection(d);
		return sqrt(d.x*d.x + d.y*d.y);
	}

	__inline__ __device__ float2 correctionIncrement(float2 pos1, float2 pos2) const
	{
		float2 result{ 0.0f, 0.0f };
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


protected:
	int2 _size;
};

template<typename T>
class Map
    : public BasicMap
{
public:
    __host__ __inline__ void init(int2 const& size, int maxEntries)
    {
        BasicMap::init(size);
        CudaMemoryManager::getInstance().acquireMemory<T*>(size.x * size.y, _map);
        _mapEntries.init(maxEntries);

        std::vector<T*> hostMap(size.x * size.y, 0);
        checkCudaErrors(cudaMemcpy(_map, hostMap.data(), sizeof(T*)*size.x*size.y, cudaMemcpyHostToDevice));
    }

    __device__ __inline__ void reset()
    {
        _mapEntries.reset();
    }

    __host__ __inline__ void free()
    {
        CudaMemoryManager::getInstance().freeMemory(_map);
        _mapEntries.free();
    }

	__device__ __inline__ bool isEntityPresent(float2 const& pos, T* entity) const
	{
		int2 posInt = { floorInt(pos.x), floorInt(pos.y) };
		mapPosCorrection(posInt);
		auto mapEntry = posInt.x + posInt.y * BasicMap::_size.x;
		return _map[mapEntry] == entity;
	}

	__device__ __inline__ T* get(float2 const& pos) const
	{
		int2 posInt = { floorInt(pos.x), floorInt(pos.y) };
		mapPosCorrection(posInt);
		auto mapEntry = posInt.x + posInt.y * _size.x;
		return _map[mapEntry];
	}

	__device__ __inline__ void set(float2 const& pos, T* entity)
	{
		int2 posInt = { floorInt(pos.x), floorInt(pos.y) };
		mapPosCorrection(posInt);
		auto mapEntry = posInt.x + posInt.y * _size.x;
        _map[mapEntry] = entity;
	}

    __device__ __inline__ void set_blockCall(int numEntities, T** entities)
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
            int2 posInt = { floorInt(entity->absPos.x), floorInt(entity->absPos.y) };
            mapPosCorrection(posInt);
            auto mapEntry = posInt.x + posInt.y * _size.x;
            _map[mapEntry] = entity;

            entrySubarray[index] = mapEntry;
        }
        __syncthreads();
    }

    __device__ __inline__ void cleanup_gridCall()
    {
        auto partition =
            calcPartition(_mapEntries.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);
        for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
            auto mapEntry = _mapEntries.at(index);
            _map[mapEntry] = nullptr;
        }
    }

private:
	T ** _map;
    Array<int> _mapEntries;
};

