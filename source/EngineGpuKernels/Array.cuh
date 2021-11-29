#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda/helper_cuda.h>

#include "CudaMemoryManager.cuh"
#include "Swap.cuh"

namespace Const
{
    constexpr float ArrayFillLevelFactor = 2.0f / 3.0f;
}

template <class T>
class Array
{
private:
    int* _size;
    int* _numEntries;

public:
    T** _data;
    Array()
    {}

    __host__ __inline__ void init()
    {
        T* data = nullptr;
        CudaMemoryManager::getInstance().acquireMemory<T*>(1, _data);
        CudaMemoryManager::getInstance().acquireMemory<int>(1, _numEntries);
        CudaMemoryManager::getInstance().acquireMemory<int>(1, _size);

        CHECK_FOR_CUDA_ERROR(cudaMemcpy(_data, &data, sizeof(T*), cudaMemcpyHostToDevice));
        CHECK_FOR_CUDA_ERROR(cudaMemset(_numEntries, 0, sizeof(int)));
        CHECK_FOR_CUDA_ERROR(cudaMemset(_size, 0, sizeof(int)));
    }


    __host__ __inline__ void init(int size)
    {
        T* data = nullptr;
        CudaMemoryManager::getInstance().acquireMemory<T>(size, data);
        CudaMemoryManager::getInstance().acquireMemory<T*>(1, _data);
        CudaMemoryManager::getInstance().acquireMemory<int>(1, _numEntries);
        CudaMemoryManager::getInstance().acquireMemory<int>(1, _size);

        CHECK_FOR_CUDA_ERROR(cudaMemcpy(_data, &data, sizeof(T*), cudaMemcpyHostToDevice));
        CHECK_FOR_CUDA_ERROR(cudaMemset(_numEntries, 0, sizeof(int)));
        CHECK_FOR_CUDA_ERROR(cudaMemset(_size, size, sizeof(int)));
    }

    __host__ __inline__ void free()
    {
        T* data = nullptr;
        CHECK_FOR_CUDA_ERROR(cudaMemcpy(&data, _data, sizeof(T*), cudaMemcpyDeviceToHost));

        CudaMemoryManager::getInstance().freeMemory(data);
        CudaMemoryManager::getInstance().freeMemory(_data);
        CudaMemoryManager::getInstance().freeMemory(_numEntries);
        CudaMemoryManager::getInstance().freeMemory(_size);
    }

    __device__ __inline__ T* getArray() const { return *_data; }
    __host__ __inline__ T* getArray_host() const
    {
        T* result;
        CHECK_FOR_CUDA_ERROR(cudaMemcpy(&result, _data, sizeof(T*), cudaMemcpyDeviceToHost));
        return result;
    }
    __host__ __inline__ void setArray_host(T* data) const
    {
        CHECK_FOR_CUDA_ERROR(cudaMemcpy(_data, &data, sizeof(T*), cudaMemcpyHostToDevice));
    }

    __device__ __inline__ int getSize() const { return *_size; }
    __host__ __inline__ int getSize_host() const
    {
        int result;
        CHECK_FOR_CUDA_ERROR(cudaMemcpy(&result, _size, sizeof(int), cudaMemcpyDeviceToHost));
        return result;
    }

    __device__ __inline__ int getNumEntries() const { return *_numEntries; }
    __device__ __inline__ void setNumEntries(int value) const { *_numEntries = value; }
    __host__ __inline__ int getNumEntries_host() const
    {
        int result;
        CHECK_FOR_CUDA_ERROR(cudaMemcpy(&result, _numEntries, sizeof(int), cudaMemcpyDeviceToHost));
        return result;
    }
    __host__ __inline__ void setNumEntries_host(int value)
    {
        checkCudaErrors(cudaMemcpy(_numEntries, &value, sizeof(int), cudaMemcpyHostToDevice));
    }

    __device__ __inline__ void swapContent(Array& other)
    {
        swap(*_numEntries, *other._numEntries);
        swap(*_data, *other._data);
    }
    __host__ __inline__ void swapContent_host(Array& other)
    {
        auto numEntries = getNumEntries_host();
        auto otherNumEntries = other.getNumEntries_host();
        setNumEntries_host(otherNumEntries);
        other.setNumEntries_host(numEntries);

        auto data = getArray_host();
        auto otherData = other.getArray_host();
        setArray_host(otherData);
        other.setArray_host(data);
    }

    __device__ __inline__ void reset() { *_numEntries = 0; }

    __device__ __inline__ T* getNewSubarray(int size)
    {
        int oldIndex = atomicAdd(_numEntries, static_cast<int>(size));
        if (oldIndex + size - 1 >= *_size) {
            atomicAdd(_numEntries, -size);
            printf("Not enough fixed memory!\n");
            ABORT();
        }
        return &(*_data)[oldIndex];
    }

    __device__ __inline__ T* getNewElement()
    {
        int oldIndex = atomicAdd(_numEntries, 1);
        if (oldIndex >= *_size) {
            atomicAdd(_numEntries, -1);
            printf("Not enough fixed memory!\n");
            ABORT();
        }
        return &(*_data)[oldIndex];
    }

    __device__ __inline__ T& at(int index) { return (*_data)[index]; }
    __device__ __inline__ T const& at(int index) const { return (*_data)[index]; }

    __device__ __inline__ int decNumEntriesAndReturnOrigSize() { return atomicSub(_numEntries, 1); }

    __device__ __inline__ bool shouldResize(int arraySizeInc) const
    {
        return getNumEntries() + arraySizeInc > getSize() * Const::ArrayFillLevelFactor;
    }
    __host__ __inline__ bool shouldResize_host(int arraySizeInc) const
    {
        return getNumEntries_host() + arraySizeInc > getSize_host() * Const::ArrayFillLevelFactor;
    }

    __host__ __inline__ void resize(int newSize) const
    {
        auto size = getSize_host();

        if (size == newSize) {
            return;
        }
        if (size > 0) {
            T* data;
            CHECK_FOR_CUDA_ERROR(cudaMemcpy(&data, _data, sizeof(T*), cudaMemcpyDeviceToHost));
            CudaMemoryManager::getInstance().freeMemory(data);
        }
        T* newData;
        CudaMemoryManager::getInstance().acquireMemory<T>(newSize, newData);
        CHECK_FOR_CUDA_ERROR(cudaMemcpy(_data, &newData, sizeof(T*), cudaMemcpyHostToDevice));
        CHECK_FOR_CUDA_ERROR(cudaMemcpy(_size, &newSize, sizeof(int), cudaMemcpyHostToDevice));
    }
};
