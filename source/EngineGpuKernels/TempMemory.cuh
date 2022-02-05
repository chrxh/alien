#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda/helper_cuda.h>

#include "CudaMemoryManager.cuh"
#include "Array.cuh"

class TempMemory
{
private:
    uint64_t _size;
    int* _bytesOccupied;
    unsigned char** _data;

public:
    TempMemory()
        : _size(0)
    {}

    __host__ __inline__ void init()
    {
        _size = 0;
        CudaMemoryManager::getInstance().acquireMemory<unsigned char*>(1, _data);
        CudaMemoryManager::getInstance().acquireMemory<int>(1, _bytesOccupied);

        CHECK_FOR_CUDA_ERROR(cudaMemset(_bytesOccupied, 0, sizeof(int)));
    }

    __host__ __inline__ void resize(uint64_t size)
    {
/*
        if (_size > 0) {
            unsigned char* data = nullptr;
            CHECK_FOR_CUDA_ERROR(cudaMemcpy(&data, _data, sizeof(unsigned char*), cudaMemcpyDeviceToHost));
            CudaMemoryManager::getInstance().freeMemory(data);
        }
*/

        _size = size;
        unsigned char* data = nullptr;
        CudaMemoryManager::getInstance().acquireMemory<unsigned char>(size, data);
        CHECK_FOR_CUDA_ERROR(cudaMemcpy(_data, &data, sizeof(unsigned char*), cudaMemcpyHostToDevice));
    }

    uint64_t getSize() const { return _size; }

    __host__ __inline__ void free()
    {

        if (_size > 0) {
            unsigned char* data = nullptr;
            CHECK_FOR_CUDA_ERROR(cudaMemcpy(&data, _data, sizeof(unsigned char*), cudaMemcpyDeviceToHost));
            CudaMemoryManager::getInstance().freeMemory(data);
        }
        CudaMemoryManager::getInstance().freeMemory(_data);
        CudaMemoryManager::getInstance().freeMemory(_bytesOccupied);
        _size = 0;
    }

    template<typename T>
    __device__ __inline__ T* getArray(int numElements)
    {
        if (0 == numElements) {
            return nullptr;
        }
        int newBytesToOccupy = numElements * sizeof(T);
        newBytesToOccupy = newBytesToOccupy + 16 - (newBytesToOccupy % 16);
        int oldIndex = atomicAdd(_bytesOccupied, newBytesToOccupy);
        if (oldIndex + newBytesToOccupy - 1 >= _size) {
            atomicAdd(_bytesOccupied, -newBytesToOccupy);
            printf("Not enough temporary memory!\n");
            ABORT();
        }
        return reinterpret_cast<T*>(&(*_data)[oldIndex]);
    }

    __host__ __inline__ unsigned char* getData_host() const
    {
        unsigned char* data;
        CHECK_FOR_CUDA_ERROR(cudaMemcpy(&data, _data, sizeof(unsigned char*), cudaMemcpyDeviceToHost));
        return data;
    }
    __host__ __inline__ void setData_host(unsigned char* data) const { CHECK_FOR_CUDA_ERROR(cudaMemcpy(_data, &data, sizeof(unsigned char*), cudaMemcpyHostToDevice)); }


    __device__ __inline__ int getNumBytes() { return *_bytesOccupied; }
    __host__ __inline__ int getNumBytes_host() {
        int result;
        CHECK_FOR_CUDA_ERROR(cudaMemcpy(&result, _bytesOccupied, sizeof(int), cudaMemcpyDeviceToHost));
        return result;
    }
    __host__ __inline__ void setNumBytes_host(int value) { checkCudaErrors(cudaMemcpy(_bytesOccupied, &value, sizeof(int), cudaMemcpyHostToDevice)); }

    __device__ __inline__ void reset() { *_bytesOccupied = 0; }

    __device__ __inline__ void swapContent(TempMemory& other)
    {
        swap(*_bytesOccupied, *other._bytesOccupied);
        swap(*_data, *other._data);
    }

    __host__ __inline__ void swapContent_host(TempMemory& other)
    {
        auto numBytes = getNumBytes_host();
        auto otherNumBytes = other.getNumBytes_host();
        setNumBytes_host(otherNumBytes);
        other.setNumBytes_host(numBytes);

        auto data = getData_host();
        auto otherData = other.getData_host();
        setData_host(otherData);
        other.setData_host(data);
    }
};
