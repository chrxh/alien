#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <helper_cuda.h>

#include "CudaMemoryManager.cuh"
#include "Swap.cuh"

template <class T>
class Array
{
private:
    int _size;
    int* _numEntries;
    T** _data;

public:
    Array()
        : _size(0)
    {}

    __host__ __inline__ void init(int size)
    {
        _size = size;
        T* data = nullptr;
        CudaMemoryManager::getInstance().acquireMemory<T>(size, data);
        CudaMemoryManager::getInstance().acquireMemory<T*>(1, _data);
        CudaMemoryManager::getInstance().acquireMemory<int>(1, _numEntries);

        checkCudaErrors(cudaMemcpy(_data, &data, sizeof(T*), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemset(_numEntries, 0, sizeof(int)));
    }

    __host__ __inline__ T* getArrayForHost() const
    {
        T* result;
        checkCudaErrors(cudaMemcpy(&result, _data, sizeof(T*), cudaMemcpyDeviceToHost));
        return result;
    }

    __device__ __inline__ T* getArrayForDevice() const { return *_data; }


    __host__ __inline__ void free()
    {
        T* data = nullptr;
        checkCudaErrors(cudaMemcpy(&data, _data, sizeof(T*), cudaMemcpyDeviceToHost));

        CudaMemoryManager::getInstance().freeMemory(data);
        CudaMemoryManager::getInstance().freeMemory(_data);
        CudaMemoryManager::getInstance().freeMemory(_numEntries);
    }

    __device__ __inline__ void swapContent(Array& other)
    {
        swap(*_numEntries, *other._numEntries);
        swap(*_data, *other._data);
    }

    __device__ __inline__ void reset() { *_numEntries = 0; }

    int retrieveNumEntries() const
    {
        int result;
        checkCudaErrors(cudaMemcpy(&result, _numEntries, sizeof(int), cudaMemcpyDeviceToHost));
        return result;
    }

    __device__ __inline__ T* getNewSubarray(int size)
    {
        int oldIndex = atomicAdd(_numEntries, size);
        if (oldIndex + size - 1 >= _size) {
            atomicAdd(_numEntries, -size);
            printf("Not enough fixed memory!\n");
            ABORT();
        }
        return &(*_data)[oldIndex];
    }

    __device__ __inline__ T* getNewElement()
    {
        int oldIndex = atomicAdd(_numEntries, 1);
        if (oldIndex >= _size) {
            atomicAdd(_numEntries, -1);
            printf("Not enough fixed memory!\n");
            ABORT();
        }
        return &(*_data)[oldIndex];
    }

    __device__ __inline__ T& at(int index) { return (*_data)[index]; }
    __device__ __inline__ T const& at(int index) const { return (*_data)[index]; }

    __device__ __inline__ int getNumEntries() const { return *_numEntries; }
    __device__ __inline__ void setNumEntries(int value) const { *_numEntries = value; }
    __device__ __inline__ int decNumEntriesAndReturnOrigSize() { return atomicSub(_numEntries, 1); }
};
