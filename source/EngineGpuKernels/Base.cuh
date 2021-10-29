#pragma once

#include <vector>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <helper_cuda.h>

#include "EngineInterface/GpuSettings.h"

#include "Array.cuh"
#include "CudaMemoryManager.cuh"
#include "Definitions.cuh"
#include "HashSet.cuh"
#include "Swap.cuh"

__device__ inline float toFloat(int value)
{
    return static_cast<float>(value);
}

__device__ inline int toInt(float value)
{
    return static_cast<int>(value);
}

struct PartitionData
{
    int startIndex;
    int endIndex;

    __inline__ __device__ int numElements() const { return endIndex - startIndex + 1; }
};

class CudaNumberGenerator
{
private:
    unsigned int* _currentIndex;
    int* _array;
    int _size;

    unsigned long long int* _currentId;

public:
    void init(int size)
    {
        _size = size;

        CudaMemoryManager::getInstance().acquireMemory<unsigned int>(1, _currentIndex);
        CudaMemoryManager::getInstance().acquireMemory<int>(size, _array);
        CudaMemoryManager::getInstance().acquireMemory<unsigned long long int>(1, _currentId);

        checkCudaErrors(cudaMemset(_currentIndex, 0, sizeof(unsigned int)));
        unsigned long long int hostCurrentId = 1;
        checkCudaErrors(cudaMemcpy(_currentId, &hostCurrentId, sizeof(_currentId), cudaMemcpyHostToDevice));

        std::vector<int> randomNumbers(size);
        for (int i = 0; i < size; ++i) {
            randomNumbers[i] = rand();
        }
        checkCudaErrors(cudaMemcpy(_array, randomNumbers.data(), sizeof(int) * size, cudaMemcpyHostToDevice));
    }


    __device__ __inline__ int random(int maxVal)
    {
        int number = getRandomNumber();
        return number % (maxVal + 1);
    }

    __device__ __inline__ float random(float maxVal)
    {
        int number = getRandomNumber();
        return maxVal * static_cast<float>(number) / RAND_MAX;
    }

    __device__ __inline__ float random()
    {
        int number = getRandomNumber();
        return static_cast<float>(number) / RAND_MAX;
    }

    __device__ __inline__ unsigned long long int createNewId_kernel() { return atomicAdd(_currentId, 1); }

    __device__ __inline__ void adaptMaxId(unsigned long long int id)
    {
        atomicMax(_currentId, id + 1);
    }

    void free()
    {
        CudaMemoryManager::getInstance().freeMemory(1, _currentIndex);
        CudaMemoryManager::getInstance().freeMemory(_size, _array);
        CudaMemoryManager::getInstance().freeMemory(1, _currentId);
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
    result.startIndex = division < remainder
        ? (entitiesByDivisions + 1) * division
        : (entitiesByDivisions + 1) * remainder + entitiesByDivisions * (division - remainder);
    result.endIndex = result.startIndex + length - 1;
    return result;
}

__host__ __device__ __inline__ int2 toInt2(float2 const& p)
{
    return {static_cast<int>(p.x), static_cast<int>(p.y)};
}

__host__ __device__ __inline__ float2 toFloat2(int2 const& p)
{
    return {static_cast<float>(p.x), static_cast<float>(p.y)};
}

__host__ __device__ __inline__ int floorInt(float v)
{
    int result = static_cast<int>(v);
    if (result > v) {
        --result;
    }
    return result;
}

__host__ __device__ __inline__ bool
isContainedInRect(int2 const& rectUpperLeft, int2 const& rectLowerRight, float2 const& pos)
{
    return pos.x >= rectUpperLeft.x && pos.x <= rectLowerRight.x
        && pos.y >= rectUpperLeft.y && pos.y <= rectLowerRight.y;
}

__host__ __device__ __inline__ bool
isContainedInRect(float2 const& rectUpperLeft, float2 const& rectLowerRight, float2 const& pos)
{
    return pos.x >= rectUpperLeft.x && pos.x <= rectLowerRight.x && pos.y >= rectUpperLeft.y
        && pos.y <= rectLowerRight.y;
}

__host__ __device__ __inline__ bool
isContainedInRect(int2 const& rectUpperLeft, int2 const& rectLowerRight, int2 const& pos, int boundary = 0)
{
    return pos.x >= rectUpperLeft.x + boundary && pos.x <= rectLowerRight.x - boundary
        && pos.y >= rectUpperLeft.y + boundary && pos.y <= rectLowerRight.y - boundary;
}

template <typename T>
__device__ __inline__ T* atomicExch(T** address, T* value)
{
    return reinterpret_cast<T*>(atomicExch(
        reinterpret_cast<unsigned long long int*>(address), reinterpret_cast<unsigned long long int>(value)));
}

/*
template<typename T>
__device__ __inline__  T* atomicExch_block(T** address, T* value)
{
    return reinterpret_cast<T*>(atomicExch_block(reinterpret_cast<unsigned long long int*>(address),
        reinterpret_cast<unsigned long long int>(value)));
}
*/

class BlockLock
{
public:
    __device__ __inline__ void init_block()
    {
        if (0 == threadIdx.x) {
            lock = 0;
        }
        __syncthreads();
    }

    __device__ __inline__ void getLock()
    {
        while (1 == atomicExch_block(&lock, 1)) {
        }
    }

    __device__ __inline__ bool tryLock() { return 0 == atomicExch_block(&lock, 1); }

    __device__ __inline__ void releaseLock() { atomicExch_block(&lock, 0); }

private:
    int lock;  //0 = unlocked, 1 = locked
};

class SystemLock
{
public:
    __device__ __inline__ void init() { lock = 0; }

    __device__ __inline__ void getLock()
    {
        while (1 == atomicExch(&lock, 1)) {
        }
    }

    __device__ __inline__ bool tryLock() { return 0 == atomicExch(&lock, 1); }

    __device__ __inline__ void releaseLock() { atomicExch(&lock, 0); }

private:
    int lock;  //0 = unlocked, 1 = locked
};

class SystemDoubleLock
{
public:
    __device__ __inline__ void init(int* lock1, int* lock2)
    {
        if (lock1 <= lock2) {
            _lock1 = lock1;
            _lock2 = lock2;
        } else {
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

    __device__ __inline__ void getLock()
    {
        while (!tryLock()) {
        };
    }

    __device__ __inline__ bool isLocked() { return _lockState1 == 0 && _lockState2 == 0; }

    __device__ __inline__ void releaseLock()
    {
        if (0 == _lockState1) {
            atomicExch(_lock1, 0);
        }
        if (0 == _lockState2) {
            atomicExch(_lock2, 0);
        }
    }

private:
    int* _lock1;
    int* _lock2;
    int _lockState1;
    int _lockState2;
};

float random(float max)
{
    return ((float)rand() / RAND_MAX) * max;
}
