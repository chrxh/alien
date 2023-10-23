#pragma once

#include <vector>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda/helper_cuda.h>

#include "EngineInterface/GpuSettings.h"

#include "CudaMemoryManager.cuh"
#include "Definitions.cuh"
#include "HashSet.cuh"
#include "Swap.cuh"

template<typename T>
__device__ __host__ inline float toFloat(T value)
{
    return static_cast<float>(value);
}

template <typename T>
__device__ __host__ inline int toInt(T value)
{
    return static_cast<int>(value);
}

template <typename T>
__device__ __host__ inline uint64_t toUInt64(T value)
{
    return static_cast<uint64_t>(value);
}

struct PartitionData
{
    int startIndex;
    int endIndex;

    __inline__ __device__ int numElements() const { return endIndex - startIndex + 1; }
};

__device__ __inline__ PartitionData calcPartition(int numEntities, int index, int numIndices)
{
    PartitionData result;
    int entitiesByDivisions = numEntities / numIndices;
    int remainder = numEntities % numIndices;

    int length = index < remainder ? entitiesByDivisions + 1 : entitiesByDivisions;
    result.startIndex = index < remainder
        ? (entitiesByDivisions + 1) * index
        : (entitiesByDivisions + 1) * remainder + entitiesByDivisions * (index - remainder);
    result.endIndex = result.startIndex + length - 1;
    return result;
}

__device__ __inline__ PartitionData calcAllThreadsPartition(int numEntities)
{
    return calcPartition(
        numEntities, threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);
}

__device__ __inline__ PartitionData calcBlockPartition(int numEntities)
{
    return calcPartition(numEntities, blockIdx.x, gridDim.x);
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
__device__ __inline__ T alienAtomicRead(T* const& address)
{
    return atomicAdd(address, 0);
        
}

// CUDA headers use "unsigned long long" for 64bit types, which
// may not be structurally equivalent to std::uint64_t
//
// Due to this, we need an ugly type casting workaround here
//
template <typename T>
__device__ __inline__ T alienAtomicAdd64(T* address, T const& value)
{
    static_assert(sizeof(unsigned long long) == sizeof(T));
    return atomicAdd(reinterpret_cast<unsigned long long*>(address), static_cast<unsigned long long>(value));
}

template <typename T>
__device__ __inline__ T alienAtomicMin64(T* address, T const& value)
{
    static_assert(sizeof(unsigned long long) == sizeof(T));
    return atomicMin(reinterpret_cast<unsigned long long*>(address), static_cast<unsigned long long>(value));
}

template <typename T>
__device__ __inline__ T* alienAtomicExch(T** address, T* value)
{
    static_assert(sizeof(unsigned long long) == sizeof(T*));
    return reinterpret_cast<T*>(atomicExch(reinterpret_cast<unsigned long long int*>(address), reinterpret_cast<unsigned long long int>(value)));
}

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

    __device__ __inline__ void init(int* lock1, int* lock2, bool reverse)
    {
        if (!reverse) {
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
        __threadfence();
        return true;
    }

    __device__ __inline__ bool isLocked() { return _lockState1 == 0 && _lockState2 == 0; }

    __device__ __inline__ void releaseLock()
    {
        __threadfence();
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

inline float random(float max)
{
    return ((float)rand() / RAND_MAX) * max;
}

template <typename T>
inline T copyToHost(T* source)
{
    T result;
    CHECK_FOR_CUDA_ERROR(cudaMemcpy(&result, source, sizeof(T), cudaMemcpyDeviceToHost));
    return result;
}

template <typename T>
inline void copyToHost(T* target, T* source, int count = 1)
{
    CHECK_FOR_CUDA_ERROR(cudaMemcpy(target, source, sizeof(T) * count, cudaMemcpyDeviceToHost));
}

template <typename T>
inline void copyToDevice(T* target, T* source, int count = 1)
{
    CHECK_FOR_CUDA_ERROR(cudaMemcpy(target, source, sizeof(T) * count, cudaMemcpyHostToDevice));
}

template <typename T>
void setValueToDevice(T* target, T const& value)
{
    CHECK_FOR_CUDA_ERROR(cudaMemcpy(target, &value, sizeof(T), cudaMemcpyHostToDevice));
}

__device__ __inline__ int calcMod(char value, int count)
{
    return static_cast<unsigned char>(value) % count;
}

template<typename Container, typename LessFunc>
__device__ __inline__ void bubbleSort(Container& container, int size, LessFunc lessFunc)
{
    bool newRound = true;
    for (int i = size - 1; i >= 1; --i) {
        if (newRound) {
            newRound = false;
            for (int j = 0; j <= i - 1; ++j) {
                if (lessFunc(container[j + 1],container[j])) {
                    auto temp = container[j];
                    container[j] = container[j + 1];
                    container[j + 1] = temp;
                    newRound = true;
                }
            }
        } else {
            break;
        }
    }
}