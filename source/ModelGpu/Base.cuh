#pragma once

#include <vector>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <helper_cuda.h>

#include "Definitions.cuh"
#include "Array.cuh"
#include "CudaConstants.h"
#include "CudaMemoryManager.cuh"
#include "HashSet.cuh"

struct PartitionData
{
    int startIndex;
    int endIndex;

    __inline__ __device__ int numElements() const
    {
        return endIndex - startIndex + 1;
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

        CudaMemoryManager::getInstance().acquireMemory<unsigned int>(1, _currentIndex);
        CudaMemoryManager::getInstance().acquireMemory<int>(size, _array);
        CudaMemoryManager::getInstance().acquireMemory<uint64_t>(1, _currentId);

        checkCudaErrors(cudaMemset(_currentIndex, 0, sizeof(unsigned int)));
        uint64_t hostCurrentId = 1;
        checkCudaErrors(cudaMemcpy(_currentId, &hostCurrentId, sizeof(uint64_t), cudaMemcpyHostToDevice));

        std::vector<int> randomNumbers(size);
        for (int i = 0; i < size; ++i) {
            randomNumbers[i] = rand();
        }
        checkCudaErrors(cudaMemcpy(_array, randomNumbers.data(), sizeof(int)*size, cudaMemcpyHostToDevice));
    }

    __device__ __inline__ float random(int maxVal)
    {
        int number = getRandomNumber();
        return number % (maxVal + 1);
    }

    __device__ __inline__ float random(float maxVal)
    {
        int number = getRandomNumber();
        return maxVal* static_cast<float>(number) / RAND_MAX;
    }

    __device__ __inline__ float random()
    {
        int number = getRandomNumber();
        return static_cast<float>(number) / RAND_MAX;
    }

    __device__ __inline__ uint64_t createNewId_kernel()
    {
        return atomicAdd(_currentId, 1);
    }

    void free()
    {
        CudaMemoryManager::getInstance().freeMemory(_currentIndex);
        CudaMemoryManager::getInstance().freeMemory(_array);
        CudaMemoryManager::getInstance().freeMemory(_currentId);

        cudaFree(_currentId);
        cudaFree(_currentIndex);
        cudaFree(_array);
    }

private:
    __device__ __inline__ int getRandomNumber()
    {
        int index = atomicInc(_currentIndex, _size - 1);
        return _array[index];
    }
};

__device__ __inline__ PartitionData calcPartition(int numEntities, int division, int numDivisions)
{
    PartitionData result;
    int entitiesByDivisions = numEntities / numDivisions;
    int remainder = numEntities % numDivisions;

    int length = division < remainder ? entitiesByDivisions + 1 : entitiesByDivisions;
    result.startIndex = division < remainder ?
        (entitiesByDivisions + 1) * division
        : (entitiesByDivisions + 1) * remainder + entitiesByDivisions * (division - remainder);
    result.endIndex = result.startIndex + length - 1;
    return result;
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

    __device__ __inline__ bool tryLock()
    {
        _lockState1 = atomicExch(_lock1, 1);
        _lockState2 = atomicExch(_lock2, 1);
        if (0 != _lockState1 || 0 != _lockState2) {
            releaseLock();
            return false;
        }
        return true;
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
