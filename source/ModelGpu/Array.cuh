#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <helper_cuda.h>

#include "CudaMemoryManager.cuh"

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

    __host__ __inline__ void free()
    {
        T* data = nullptr;
        checkCudaErrors(cudaMemcpy(&data, _data, sizeof(T*), cudaMemcpyDeviceToHost));

        CudaMemoryManager::getInstance().freeMemory(data);
        CudaMemoryManager::getInstance().freeMemory(_data);
        CudaMemoryManager::getInstance().freeMemory(_numEntries);
    }

    __device__ __inline__ void swapArray(Array& other)
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
            printf("Not enough memory [Array]!\n");
            return nullptr;
        }
        return &(*_data)[oldIndex];
    }

    __device__ __inline__ T* getNewElement()
    {
        int oldIndex = atomicAdd(_numEntries, 1);
        if (oldIndex >= _size) {
            atomicAdd(_numEntries, -1);
            printf("Not enough memory [Array]!\n");
            return nullptr;
        }
        return &(*_data)[oldIndex];
    }

    __device__ __inline__ T& at(int index) { return (*_data)[index]; }
    __device__ __inline__ T const& at(int index) const { return (*_data)[index]; }

    __device__ __inline__ int getNumEntries() const { return *_numEntries; }
    __device__ __inline__ void setNumEntries(int value) const { *_numEntries = value; }

    __device__ __inline__ T* getEntireArray() const { return *_data; }

};

class DynamicMemory
{
private:
    int _size;
    int* _bytesOccupied;
    unsigned char** _data;

public:
    DynamicMemory()
        : _size(0)
    {}

    __host__ __inline__ void init(uint64_t size)
    {
        _size = size;
        unsigned char* data = nullptr;
        CudaMemoryManager::getInstance().acquireMemory<unsigned char>(size, data);
        CudaMemoryManager::getInstance().acquireMemory<unsigned char*>(1, _data);
        CudaMemoryManager::getInstance().acquireMemory<int>(1, _bytesOccupied);

        checkCudaErrors(cudaMemcpy(_data, &data, sizeof(unsigned char*), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemset(_bytesOccupied, 0, sizeof(int)));
    }

    __host__ __inline__ void free()
    {
        unsigned char* data = nullptr;
        checkCudaErrors(cudaMemcpy(&data, _data, sizeof(unsigned char*), cudaMemcpyDeviceToHost));

        CudaMemoryManager::getInstance().freeMemory(data);
        CudaMemoryManager::getInstance().freeMemory(_data);
        CudaMemoryManager::getInstance().freeMemory(_bytesOccupied);
    }

    template<typename T>
    __device__ __inline__ T* getArray (int numElements)
    {
        int newBytesToOccupy = numElements * sizeof(T);
        newBytesToOccupy = newBytesToOccupy + 16 - (newBytesToOccupy % 16);
        int oldIndex = atomicAdd(_bytesOccupied, newBytesToOccupy);
        if (oldIndex + newBytesToOccupy - 1 >= _size) {
            atomicAdd(_bytesOccupied, -newBytesToOccupy);
            printf("Not enough memory [ArrayController]!\n");
            return nullptr;
        }
        return reinterpret_cast<T*>(&(*_data)[oldIndex]);
    }

    __device__ __inline__ void reset() { *_bytesOccupied = 0; }
};