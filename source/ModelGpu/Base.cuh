#pragma once

#include <vector>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <helper_cuda.h>

#include "CudaConstants.cuh"


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

		checkCudaErrors(cudaMalloc(&_currentIndex, sizeof(unsigned int)));
		checkCudaErrors(cudaMalloc(&_array, sizeof(int)*size));
		checkCudaErrors(cudaMalloc(&_currentId, sizeof(uint64_t)));

		checkCudaErrors(cudaMemset(_currentIndex, 0, sizeof(unsigned int)));
		uint64_t hostCurrentId = 0;
		checkCudaErrors(cudaMemcpy(_currentId, &hostCurrentId, sizeof(uint64_t), cudaMemcpyHostToDevice));

		std::vector<int> randomNumbers(size);
		for (int i = 0; i < size; ++i) {
			randomNumbers[i] = rand();
		}
		checkCudaErrors(cudaMemcpy(_array, randomNumbers.data(), sizeof(int)*size, cudaMemcpyHostToDevice));
	}

	__device__ __inline__ float random(int maxVal)
	{
		int index = atomicInc(_currentIndex, _size - 1);
		int number = _array[index];
		return number % (maxVal + 1);
	}

	__device__ __inline__ float random(float maxVal)
	{
		int index = atomicInc(_currentIndex, _size - 1);
		int number = _array[index];
		return maxVal* static_cast<float>(number) / RAND_MAX;
	}

	__device__ __inline__ float random()
	{
		int index = atomicInc(_currentIndex, _size - 1);
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
		checkCudaErrors(cudaMalloc(&_numEntries, sizeof(int)));
		checkCudaErrors(cudaMemset(_numEntries, 0, sizeof(int)));
	}

	ArrayController(int size)
		: _size(size)
	{
		checkCudaErrors(cudaMalloc(&_data, sizeof(T) * size));
		checkCudaErrors(cudaMalloc(&_numEntries, sizeof(int)));
		checkCudaErrors(cudaMemset(_numEntries, 0, sizeof(int)));
	}

	void free()
	{
		checkCudaErrors(cudaFree(_data));
		checkCudaErrors(cudaFree(_numEntries));
	}

	void reset()
	{
		checkCudaErrors(cudaMemset(_numEntries, 0, sizeof(int)));
	}

    int retrieveNumEntries() const
    {
        int result;
        checkCudaErrors(cudaMemcpy(&result, _numEntries, sizeof(int), cudaMemcpyDeviceToHost));
        return result;
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

	__device__ __inline__ T* at(int index)
	{
		return &_data[index];
	}

	__device__ __inline__ int getNumEntries() const
	{
		return *_numEntries;
	}

	__device__ __inline__ T* getEntireArray() const
	{
		return _data;
	}

	__device__ __inline__ void setNumEntries(int value) const
	{
		*_numEntries = value;
	}
};

template<typename T>
struct HashFunctor
{
};


template<typename T>
struct HashFunctor<T*>
{
    __device__ __inline__ int operator()(T* const& element)
    {
        return reinterpret_cast<std::uintptr_t>(element) * 17;
    }
};


template<typename T, typename Hash = HashFunctor<T>>
class HashSet
{
public:
    __device__ __inline__ HashSet(int size, T* data)
        : _size(size), _data(data)
    {
        for (int i = 0; i < size; ++i) {
            _data[i] = nullptr;
        }
    }

    __device__ __inline__ void insert(T const& element)
    {
        int index = _hash(element) % _size;
        while (_data[index]) {
            if (_data[index] == element) {
                return;
            }
            index = (++index) % _size;
        }
        _data[index] = element;
    }

    __device__ __inline__ bool contains(T const& element)
    {
        int index = _hash(element) % _size;
        for (int i = 0; i < _size; ++i, index = (++index) % _size) {
            if (_data[index] == element) {
                return true;
            }
        }
        return false;
    }

private:
    T* _data;
    int _size;
    Hash _hash;
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

__host__ __device__ __inline__ int2 toInt2(float2 const &p)
{
	return{ static_cast<int>(p.x), static_cast<int>(p.y) };
}

__host__ __device__ __inline__ int floorInt(float v)
{
	int result = static_cast<int>(v);
	if (result > v) {
		--result;
	}
	return result;
}

__host__ __device__ __inline__ bool isContained(int2 const& rectUpperLeft, int2 const& rectLowerRight, float2 const& pos)
{
	return pos.x >= rectUpperLeft.x
		&& pos.x <= rectLowerRight.x
		&& pos.y >= rectUpperLeft.y
		&& pos.y <= rectLowerRight.y;
}


class DoubleLock
{
private:
	int* _lock1;
	int* _lock2;
	int _lockState1;
	int _lockState2;

public:
	__device__ __inline__ void init(int* lock1, int* lock2)
	{
		if (lock1 <= lock2) {
			_lock1 = lock1;
			_lock2 = lock2;
		}
		else {
			_lock1 = lock2;
			_lock2 = lock1;
		}
	}

	__device__ __inline__ void tryLock()
	{
		_lockState1 = atomicExch(_lock1, 1);
		_lockState2 = atomicExch(_lock2, 1);
		if (0 != _lockState1 || 0 != _lockState2) {
			releaseLock();
		}
	}

	__device__ __inline__ bool isLocked()
	{
		return _lockState1 == 0 && _lockState2 == 0;
	}

	__device__ __inline__ void releaseLock()
	{
		if (0 == _lockState1) {
			atomicExch(_lock1, 0);
		}
		if (0 == _lockState2) {
			atomicExch(_lock2, 0);
		}
	}
};

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
