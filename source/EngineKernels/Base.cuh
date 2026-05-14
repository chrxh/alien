#pragma once

#include <vector>

#include <cuda_fp16.h>

#include <EngineInterface/CudaSettings.h>
#include <EngineInterface/EngineConstants.h>

#include "Definitions.cuh"
#include "HashSet.cuh"
#include "Macros.cuh"
#include "Util.cuh"

template <typename T>
__device__ __host__ inline float toFloat(T value)
{
    return static_cast<float>(value);
}

template <typename T>
__device__ __host__ inline double toDouble(T value)
{
    return static_cast<double>(value);
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
    int step;

    __inline__ __device__ int numElements() const
    {
        if (startIndex > endIndex) {
            return 0;
        }
        return (endIndex - startIndex) / step + 1;
    }
};

__device__ __inline__ PartitionData calcSystemThreadPartition(uint64_t numEntities)
{
    return PartitionData{.startIndex = toInt(blockIdx.x * blockDim.x + threadIdx.x), .endIndex = toInt(numEntities - 1), .step = toInt(blockDim.x * gridDim.x)};
}

__device__ __inline__ PartitionData calcBlockPartition(uint64_t numEntities)
{
    PartitionData result;
    int entitiesByDivisions = numEntities / gridDim.x;
    int remainder = numEntities % gridDim.x;

    int length = blockIdx.x < remainder ? entitiesByDivisions + 1 : entitiesByDivisions;
    result.startIndex = blockIdx.x < remainder ? (entitiesByDivisions + 1) * blockIdx.x
                                               : (entitiesByDivisions + 1) * remainder + entitiesByDivisions * (blockIdx.x - remainder);
    result.endIndex = result.startIndex + length - 1;
    result.step = 1;
    return result;
}

__device__ __inline__ PartitionData calcThreadBlockPartition(uint64_t numEntities)
{
    PartitionData result;
    int entitiesByDivisions = numEntities / blockDim.x;
    int remainder = numEntities % blockDim.x;

    int length = threadIdx.x < remainder ? entitiesByDivisions + 1 : entitiesByDivisions;
    result.startIndex = threadIdx.x < remainder ? (entitiesByDivisions + 1) * threadIdx.x
                                                : (entitiesByDivisions + 1) * remainder + entitiesByDivisions * (threadIdx.x - remainder);
    result.endIndex = result.startIndex + length - 1;
    result.step = 1;
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

__host__ __device__ __inline__ bool isContainedInRect(int2 const& rectUpperLeft, int2 const& rectLowerRight, float2 const& pos)
{
    return pos.x >= rectUpperLeft.x && pos.x <= rectLowerRight.x && pos.y >= rectUpperLeft.y && pos.y <= rectLowerRight.y;
}

__host__ __device__ __inline__ bool isContainedInRect(float2 const& rectUpperLeft, float2 const& rectLowerRight, float2 const& pos)
{
    return pos.x >= rectUpperLeft.x && pos.x <= rectLowerRight.x && pos.y >= rectUpperLeft.y && pos.y <= rectLowerRight.y;
}

__host__ __device__ __inline__ bool isContainedInRect(int2 const& rectUpperLeft, int2 const& rectLowerRight, int2 const& pos, int boundary = 0)
{
    return pos.x >= rectUpperLeft.x + boundary && pos.x <= rectLowerRight.x - boundary && pos.y >= rectUpperLeft.y + boundary
        && pos.y <= rectLowerRight.y - boundary;
}

template <typename T>
__device__ __inline__ T alienAtomicRead(T* const& address)
{
    return atomicAdd(address, 0);
}

template <typename T>
__device__ __inline__ T* alienAtomicRead(T** const& address)
{
    static_assert(sizeof(unsigned long long) == sizeof(T*));
    return reinterpret_cast<T*>(atomicAdd(reinterpret_cast<unsigned long long int*>(address), 0ull));
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
    return static_cast<T>(atomicAdd(reinterpret_cast<unsigned long long*>(address), static_cast<unsigned long long>(value)));
}

template <typename T>
__device__ __inline__ T alienAtomicAdd32(T* address, T value)
{
    static_assert(sizeof(unsigned int) == sizeof(T));
    return reinterpret_cast<T>(atomicAdd(reinterpret_cast<unsigned int*>(address), reinterpret_cast<unsigned int>(value)));
}

template <typename T>
__device__ __inline__ T alienAtomicMin64(T* address, T const& value)
{
    static_assert(sizeof(unsigned long long) == sizeof(T));
    return static_cast<T>(atomicMin(reinterpret_cast<unsigned long long*>(address), static_cast<unsigned long long>(value)));
}

template <typename T>
__device__ __inline__ T alienAtomicMax64(T* address, T const& value)
{
    static_assert(sizeof(unsigned long long) == sizeof(T));
    return static_cast<T>(atomicMax(reinterpret_cast<unsigned long long*>(address), static_cast<unsigned long long>(value)));
}

template <typename T>
__device__ __inline__ T alienAtomicMax32(T* address, T const& value)
{
    static_assert(sizeof(unsigned int) == sizeof(T));
    return static_cast<T>(atomicMax(reinterpret_cast<unsigned int*>(address), static_cast<unsigned int>(value)));
}

template <typename T>
__device__ __inline__ T alienAtomicExch64(T* address, T const& value)
{
    static_assert(sizeof(unsigned long long) == sizeof(T));
    return static_cast<T>(atomicExch(reinterpret_cast<unsigned long long*>(address), static_cast<unsigned long long>(value)));
}

template <typename T>
__device__ __inline__ T* alienAtomicExch(T** address, T* value)
{
    static_assert(sizeof(unsigned long long) == sizeof(T*));
    return reinterpret_cast<T*>(atomicExch(reinterpret_cast<unsigned long long int*>(address), reinterpret_cast<unsigned long long int>(value)));
}

__device__ __forceinline__ float alienAtomicMax(float* addr, float value)
{
    float old;
    old = !signbit(value) ? __int_as_float(atomicMax((int*)addr, __float_as_int(value)))
                          : __uint_as_float(atomicMin((unsigned int*)addr, __float_as_uint(value)));

    return old;
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

template <typename Container, typename LessFunc>
__device__ __inline__ void bubbleSort(Container& container, int size, LessFunc lessFunc)
{
    bool newRound = true;
    for (int i = size - 1; i >= 1; --i) {
        if (newRound) {
            newRound = false;
            for (int j = 0; j <= i - 1; ++j) {
                if (lessFunc(container[j + 1], container[j])) {
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

// Efficient channel copy using float4 vectorized operations
// Copies MAX_CHANNELS (16) floats from src to dst using 4x float4 operations
__device__ __forceinline__ void copyChannels(float* __restrict__ dst, float const* __restrict__ src)
{
    static_assert(NEURONS_PER_CELL == 16, "copyChannels requires MAX_CHANNELS == 16 for float4 optimization");
    float4 const* src4 = reinterpret_cast<float4 const*>(src);
    float4* dst4 = reinterpret_cast<float4*>(dst);
    dst4[0] = src4[0];
    dst4[1] = src4[1];
    dst4[2] = src4[2];
    dst4[3] = src4[3];
}