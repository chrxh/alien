#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <helper_cuda.h>

#include "Constants.cuh"


#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600

#else

static __inline__ __device__ double atomicAdd(double *address, double val) {
	unsigned long long int* address_as_ull = (unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;
	if (val == 0.0)
		return __longlong_as_double(old);
	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
	} while (assumed != old);
	return __longlong_as_double(old);
}

#endif


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

class CudaNumberGenerator
{
private:
	unsigned int *_currentIndex;
	int *_array;
	int _size;

	uint64_t *_currentId;

public:

	void init(int size)
	{
		_size = size;
		int device;
		cudaDeviceProp prop;
		cudaGetDevice(&device);
		cudaGetDeviceProperties(&prop, device);

		cudaMallocManaged(&_currentIndex, sizeof(unsigned int));
		cudaMallocManaged(&_array, sizeof(int)*size);
		cudaMallocManaged(&_currentId, sizeof(uint64_t));
		checkCudaErrors(cudaGetLastError());

		*_currentIndex = 0;

		for (int i = 0; i < size; ++i) {
			_array[i] = rand();
		}
		*_currentId = 0;
	}

	__device__ __inline__ float random(float maxVal)
	{
		int index = atomicInc(_currentIndex, _size);
		int number = _array[index];
		return maxVal* static_cast<float>(number) / RAND_MAX;
	}

	__device__ __inline__ float random()
	{
		int index = atomicInc(_currentIndex, _size);
		int number = _array[index];
		return static_cast<float>(number) / RAND_MAX;
	}

	__host__ __inline__ uint64_t createNewId()
	{
		return (*_currentId)++;
	}

	__device__ __inline__ uint64_t createNewId_kernel()
	{
		return atomicAdd(_currentId, 1);
	}

	void free()
	{
		cudaFree(_currentId);
		cudaFree(_currentIndex);
		cudaFree(_array);
	}
};

template<class T>
class ArrayController
{
private:
	int _size;
	int *_numEntries;
	T* _data;

public:

	ArrayController()
		: _size(0)
	{
		cudaMallocManaged(&_numEntries, sizeof(int));
		checkCudaErrors(cudaGetLastError());

		*_numEntries = 0;
	}

	ArrayController(int size)
		: _size(size)
	{
		cudaMallocManaged(&_data, sizeof(T) * size);
		cudaMallocManaged(&_numEntries, sizeof(int));
		checkCudaErrors(cudaGetLastError());

		*_numEntries = 0;
	}

	void free()
	{
		cudaFree(_data);
		cudaFree(_numEntries);
	}

	__host__ __device__ __inline__ void reset()
	{
		*_numEntries = 0;
	}

	T* getArray(int size)
	{
		int oldIndex = *_numEntries;
		*_numEntries += size;
		return &_data[oldIndex];
	}

	__device__ __inline__ T* getNewSubarray(int size)
	{
		int oldIndex = atomicAdd(_numEntries, size);
		return &_data[oldIndex];
	}

	__device__ __inline__ T* getNewElement()
	{
		int oldIndex = atomicAdd(_numEntries, 1);
		return &_data[oldIndex];
	}

	__host__ __device__ __inline__ T* at(int index)
	{
		return &_data[index];
	}

	__host__ __device__ __inline__ int getNumEntries() const
	{
		return *_numEntries;
	}

	__host__ __device__ __inline__ T* getEntireArray() const
	{
		return _data;
	}

	__host__ __device__ __inline__ void setNumEntries(int value) const
	{
		*_numEntries = value;
	}
};

__device__ __inline__ void calcPartition(int numEntities, int division, int numDivisions, int& startIndex, int& endIndex)
{
	int entitiesByDivisions = numEntities / numDivisions;
	int remainder = numEntities % numDivisions;

	int length = division < remainder ? entitiesByDivisions + 1 : entitiesByDivisions;
	startIndex = division < remainder ?
		(entitiesByDivisions + 1) * division
		: (entitiesByDivisions + 1) * remainder + entitiesByDivisions * (division - remainder);
	endIndex = startIndex + length - 1;
}

__device__ __inline__ void normalize(float2 &vec)
{
	float length = sqrt(vec.x*vec.x + vec.y*vec.y);
	if (length > FP_PRECISION) {
		vec = { vec.x / length, vec.y / length };
	}
	else
	{
		vec = { 1.0, 0.0 };
	}
}

__host__ __device__ __inline__ float dot(float2 const &p, float2 const &q)
{
	return p.x*q.x + p.y*q.y;
}

__host__ __device__ __inline__ float2 minus(float2 const &p)
{
	return{ -p.x, -p.y };
}

__host__ __device__ __inline__ float2 mul(float2 const &p, float r)
{
	return{ p.x * r, p.y * r };
}

__host__ __device__ __inline__ float2 div(float2 const &p, float r)
{
	return{ p.x / r, p.y / r };
}

__host__ __device__ __inline__ float2 add(float2 const &p, float2 const &q)
{
	return{ p.x + q.x, p.y + q.y };
}

__host__ __device__ __inline__ float2 sub(float2 const &p, float2 const &q)
{
	return{ p.x - q.x, p.y - q.y };
}

__host__ __device__ __inline__ int2 toInt2(float2 const &p)
{
	return{ static_cast<int>(p.x), static_cast<int>(p.y) };
}

float random(float max)
{
	return ((float)rand() / RAND_MAX) * max;
}

template<typename T>
__host__ __device__ __inline__ void swap(T &a, T &b)
{
	T temp = a;
	a = b;
	b = temp;
}
