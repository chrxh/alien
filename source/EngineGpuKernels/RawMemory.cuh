#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda/helper_cuda.h>

#include "CudaMemoryManager.cuh"
#include "Array.cuh"

class RawMemory
{
private:
    uint64_t _size;
    uint64_t* _bytesOccupied;
    uint8_t** _data;

public:
    RawMemory()
        : _size(0)
    {}

    __host__ __inline__ void init()
    {
        _size = 0;
        CudaMemoryManager::getInstance().acquireMemory<uint8_t*>(1, _data);
        CudaMemoryManager::getInstance().acquireMemory<uint64_t>(1, _bytesOccupied);

        CHECK_FOR_CUDA_ERROR(cudaMemset(_bytesOccupied, 0, sizeof(uint64_t)));
    }

    __host__ __inline__ void resize(uint64_t size)
    {
        if (_size > 0) {
            uint8_t* data = nullptr;
            CHECK_FOR_CUDA_ERROR(cudaMemcpy(&data, _data, sizeof(uint8_t*), cudaMemcpyDeviceToHost));
            CudaMemoryManager::getInstance().freeMemory(data);
        }

        _size = size;
        uint8_t* data = nullptr;
        CudaMemoryManager::getInstance().acquireMemory<uint8_t>(size, data);
        CHECK_FOR_CUDA_ERROR(cudaMemcpy(_data, &data, sizeof(uint8_t*), cudaMemcpyHostToDevice));
    }

    uint64_t getSize() const { return _size; }

    __host__ __inline__ void free()
    {

        if (_size > 0) {
            uint8_t* data = nullptr;
            CHECK_FOR_CUDA_ERROR(cudaMemcpy(&data, _data, sizeof(uint8_t*), cudaMemcpyDeviceToHost));
            CudaMemoryManager::getInstance().freeMemory(data);
        }
        CudaMemoryManager::getInstance().freeMemory(_data);
        CudaMemoryManager::getInstance().freeMemory(_bytesOccupied);
        _size = 0;
    }

    template<typename T>
    __device__ __inline__ T* getArray(uint64_t numElements)
    {
        if (0 == numElements) {
            return nullptr;
        }
        uint64_t newBytesToOccupy = numElements * sizeof(T);
        newBytesToOccupy = newBytesToOccupy + 16 - (newBytesToOccupy % 16);
        uint64_t oldIndex = atomicAdd(_bytesOccupied, newBytesToOccupy);
        if (oldIndex + newBytesToOccupy - 1 >= _size) {
            atomicAdd(_bytesOccupied, (unsigned long long int)(-newBytesToOccupy));
            printf("Not enough temporary memory!\n");
            ABORT();
        }
        return reinterpret_cast<T*>(&(*_data)[oldIndex]);
    }

    __host__ __inline__ uint8_t* getData_host() const
    {
        uint8_t* data;
        CHECK_FOR_CUDA_ERROR(cudaMemcpy(&data, _data, sizeof(uint8_t*), cudaMemcpyDeviceToHost));
        return data;
    }
    __host__ __inline__ void setData_host(uint8_t* data) const { CHECK_FOR_CUDA_ERROR(cudaMemcpy(_data, &data, sizeof(uint8_t*), cudaMemcpyHostToDevice)); }


    __device__ __inline__ uint64_t getNumBytes() { return *_bytesOccupied; }
    __host__ __inline__ uint64_t getNumBytes_host()
    {
        uint64_t result;
        CHECK_FOR_CUDA_ERROR(cudaMemcpy(&result, _bytesOccupied, sizeof(uint64_t), cudaMemcpyDeviceToHost));
        return result;
    }
    __host__ __inline__ void setNumBytes_host(uint64_t value) { checkCudaErrors(cudaMemcpy(_bytesOccupied, &value, sizeof(uint64_t), cudaMemcpyHostToDevice)); }

    __device__ __inline__ void reset() { *_bytesOccupied = 0; }

    __device__ __inline__ void swapContent(RawMemory& other)
    {
        swap(*_bytesOccupied, *other._bytesOccupied);
        swap(*_data, *other._data);
    }
};
