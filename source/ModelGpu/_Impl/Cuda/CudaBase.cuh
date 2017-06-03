#pragma once

#include <cuda_runtime.h>

#define DEG_TO_RAD 3.1415926535897932384626433832795/180.0

double random(double max)
{
	return ((double)rand() / RAND_MAX) * max;
}

template<class T>
struct SharedMemory
{
	__device__ inline operator T *()
	{
		extern __shared__ int __smem[];
		return (T *)__smem;
	}

	__device__ inline operator const T *() const
	{
		extern __shared__ int __smem[];
		return (T *)__smem;
	}
};

template<class T>
class ArrayController
{
private:
	int _size;
	int _lastEntry = 0;
	T* _data;

public:

	ArrayController() = default;
	ArrayController(int size)
		: _size(size)
	{
		cudaMallocManaged(&_data, sizeof(T) * size);
	}

	void free()
	{
		cudaFree(_data);
	}

	T* getArray(int size)
	{
		auto result = _lastEntry;
		_lastEntry += size;
		return &_data[result];
	}

	__device__ T* getArrayKernel(int size)
	{
		auto result = _lastEntry;
		_lastEntry += size;
		return &_data[result];
	}

	__device__ T* getElementKernel()
	{
		return &_data[_lastEntry++];
	}

	__device__ T* getDataKernel() const
	{
		return _data;
	}
};

__device__ void tiling_Kernel(int numEntities, int division, int numDivisions, int& startIndex, int& endIndex)
{
	int entitiesByDivisions = numEntities / numDivisions;
	int remainder = numEntities % numDivisions;

	int length = division < remainder ? entitiesByDivisions + 1 : entitiesByDivisions;
	startIndex = division < remainder ?
		(entitiesByDivisions + 1) * division
		: (entitiesByDivisions + 1) * remainder + entitiesByDivisions * (division - remainder);
	endIndex = startIndex + length - 1;
}
