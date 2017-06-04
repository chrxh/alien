#pragma once

#include <cuda_runtime.h>

#define DEG_TO_RAD 3.1415926535897932384626433832795/180.0

double random(double max)
{
	return ((double)rand() / RAND_MAX) * max;
}

template<typename T>
void swap(T &a, T &b)
{
	T temp = a;
	a = b;
	b = temp;
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
	int *_numEntries = nullptr;
	T* _data;

public:

	ArrayController() = default;
	ArrayController(int size)
		: _size(size)
	{
		cudaMallocManaged(&_data, sizeof(T) * size);
		cudaMallocManaged(&_numEntries, sizeof(int));
	}

	void free()
	{
		cudaFree(_data);
		cudaFree(_numEntries);
	}

	void reset()
	{
		*_numEntries = 0;
	}
	
	T* getArray(int size)
	{
		auto result = *_numEntries;
		*_numEntries += size;
		return &_data[result];
	}

	int getNumEntries() const
	{
		return *_numEntries;
	}

	T* getEntireArray() const
	{
		return _data;
	}

	__device__ T* getArrayKernel(int size)
	{
		int oldIndex = atomicAdd(_numEntries, size);
		return &_data[oldIndex];
	}

	__device__ T* getElementKernel()
	{
		int oldIndex = atomicAdd(_numEntries, 1);
		return &_data[oldIndex];
	}

	__device__ int getNumEntriesKernel() const
	{
		return *_numEntries;
	}

	__device__ T* getEntireArrayKernel() const
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
