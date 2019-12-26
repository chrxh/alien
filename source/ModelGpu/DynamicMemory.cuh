#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <helper_cuda.h>

#include "CudaMemoryManager.cuh"

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
    __device__ __inline__ T* getArray(int numElements)
    {
        int newBytesToOccupy = numElements * sizeof(T);
        newBytesToOccupy = newBytesToOccupy + 16 - (newBytesToOccupy % 16);
        int oldIndex = atomicAdd(_bytesOccupied, newBytesToOccupy);
        if (oldIndex + newBytesToOccupy - 1 >= _size) {
            atomicAdd(_bytesOccupied, -newBytesToOccupy);
            printf("Not enough dynamic memory!\n");
            return nullptr;
        }
        return reinterpret_cast<T*>(&(*_data)[oldIndex]);
    }

    __device__ __inline__ void reset() { *_bytesOccupied = 0; }
};