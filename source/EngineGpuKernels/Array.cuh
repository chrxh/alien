#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda/helper_cuda.h>

#include "CudaMemoryManager.cuh"
#include "Util.cuh"
#include "Base.cuh"

namespace Const
{
    constexpr float ArrayFillLevelFactor = 2.0f / 4.0f;
}

template <typename T>
class Array
{
protected:
    uint64_t* _size;
    uint64_t* _numEntries;
    uint64_t* _numOrigEntries;

public:
    T** _data;
    Array() {}

    //methods for host
    __host__ __inline__ void init()
    {
        T* data = nullptr;
        CudaMemoryManager::getInstance().acquireMemory<T*>(1, _data);
        CudaMemoryManager::getInstance().acquireMemory<uint64_t>(1, _numEntries);
        CudaMemoryManager::getInstance().acquireMemory<uint64_t>(1, _numOrigEntries);
        CudaMemoryManager::getInstance().acquireMemory<uint64_t>(1, _size);

        CHECK_FOR_CUDA_ERROR(cudaMemcpy(_data, &data, sizeof(T*), cudaMemcpyHostToDevice));
        CHECK_FOR_CUDA_ERROR(cudaMemset(_numEntries, 0, sizeof(uint64_t)));
        CHECK_FOR_CUDA_ERROR(cudaMemset(_numOrigEntries, 0, sizeof(uint64_t)));
        CHECK_FOR_CUDA_ERROR(cudaMemset(_size, 0, sizeof(uint64_t)));
    }

    __host__ __inline__ void init(uint64_t size)
    {
        T* data = nullptr;
        CudaMemoryManager::getInstance().acquireMemory<T>(size, data);
        CudaMemoryManager::getInstance().acquireMemory<T*>(1, _data);
        CudaMemoryManager::getInstance().acquireMemory<uint64_t>(1, _numEntries);
        CudaMemoryManager::getInstance().acquireMemory<uint64_t>(1, _numOrigEntries);
        CudaMemoryManager::getInstance().acquireMemory<uint64_t>(1, _size);

        CHECK_FOR_CUDA_ERROR(cudaMemcpy(_data, &data, sizeof(T*), cudaMemcpyHostToDevice));
        CHECK_FOR_CUDA_ERROR(cudaMemset(_numEntries, 0, sizeof(uint64_t)));
        CHECK_FOR_CUDA_ERROR(cudaMemset(_numOrigEntries, 0, sizeof(uint64_t)));
        CHECK_FOR_CUDA_ERROR(cudaMemset(_size, size, sizeof(uint64_t)));
    }

    __host__ __inline__ void free()
    {
        T* data = nullptr;
        cudaMemcpy(&data, _data, sizeof(T*), cudaMemcpyDeviceToHost);

        CudaMemoryManager::getInstance().freeMemory(data);
        CudaMemoryManager::getInstance().freeMemory(_data);
        CudaMemoryManager::getInstance().freeMemory(_numEntries);
        CudaMemoryManager::getInstance().freeMemory(_numOrigEntries);
        CudaMemoryManager::getInstance().freeMemory(_size);
    }

    __host__ __inline__ void resize(uint64_t newSize) const
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
        CHECK_FOR_CUDA_ERROR(cudaMemcpy(_size, &newSize, sizeof(uint64_t), cudaMemcpyHostToDevice));
    }

    __host__ __inline__ T* getArray_host() const
    {
        T* result;
        CHECK_FOR_CUDA_ERROR(cudaMemcpy(&result, _data, sizeof(T*), cudaMemcpyDeviceToHost));
        return result;
    }
    __host__ __inline__ void setArray_host(T* data) const { CHECK_FOR_CUDA_ERROR(cudaMemcpy(_data, &data, sizeof(T*), cudaMemcpyHostToDevice)); }

    __host__ __inline__ uint64_t getSize_host() const
    {
        uint64_t result;
        CHECK_FOR_CUDA_ERROR(cudaMemcpy(&result, _size, sizeof(uint64_t), cudaMemcpyDeviceToHost));
        return result;
    }

    __host__ __inline__ uint64_t getNumEntries_host() const
    {
        uint64_t result;
        CHECK_FOR_CUDA_ERROR(cudaMemcpy(&result, _numEntries, sizeof(uint64_t), cudaMemcpyDeviceToHost));
        return result;
    }
    __host__ __inline__ void setNumEntries_host(uint64_t value) { checkCudaErrors(cudaMemcpy(_numEntries, &value, sizeof(uint64_t), cudaMemcpyHostToDevice)); }

    __host__ __inline__ bool shouldResize_host(uint64_t arraySizeInc) const
    {
        return getNumEntries_host() + arraySizeInc > getSize_host() * Const::ArrayFillLevelFactor;
    }

    //methods for device
    __device__ __inline__ T* getArray() const { return *_data; }

    __device__ __inline__ uint64_t getSize() const { return *_size; }

    __device__ __inline__ uint64_t getNumEntries() const { return *_numEntries; }
    __device__ __inline__ void setNumEntries(uint64_t value) const { *_numEntries = value; }
    __device__ __inline__ uint64_t getNumOrigEntries() const { return *_numOrigEntries; }
    __device__ __inline__ uint64_t saveNumEntries() const { return *_numOrigEntries = *_numEntries; }

    __device__ __inline__ void swapContent(Array& other)
    {
        swapValues(*_numEntries, *other._numEntries);
        swapValues(*_data, *other._data);
    }

    __device__ __inline__ void reset() { *_numEntries = 0; }

    __device__ __inline__ T* getSubArray(uint64_t size)
    {
        uint64_t oldIndex = alienAtomicAdd64(_numEntries, static_cast<uint64_t>(size));
        if (oldIndex + size - 1 >= *_size) {
            alienAtomicAdd64(_numEntries, -size);
            printf("Not enough fixed memory! Acquired size: %llu, total size: %llu\n", size, oldIndex + size - 1);
            ABORT();
        }
        return &(*_data)[oldIndex];
    }

    __device__ __inline__ T* getNewElement(uint64_t* index = nullptr)
    {
        uint64_t oldIndex = alienAtomicAdd64(_numEntries, uint64_t(1));
        if (oldIndex >= *_size) {
            alienAtomicAdd64(_numEntries, uint64_t(-1));
            printf("Not enough fixed memory!\n");
            ABORT();
        }
        if (index) {
            *index = oldIndex;
        }
        return &(*_data)[oldIndex];
    }

    __device__ __inline__ T& at(uint64_t index) { return (*_data)[index]; }
    __device__ __inline__ T const& at(uint64_t index) const { return (*_data)[index]; }

    __device__ __inline__ uint64_t decNumEntriesAndReturnOrigSize() { return alienAtomicAdd64(_numEntries, uint64_t(-1)); }

    __device__ __inline__ bool shouldResize(uint64_t arraySizeInc) const
    {
        return getNumEntries() + arraySizeInc > getSize() * Const::ArrayFillLevelFactor;
    }
};

class RawMemory : public Array<uint8_t>
{
public:
    template <typename T>
    __device__ __inline__ T* getTypedSubArray(uint64_t numElements)
    {
        if (0 == numElements) {
            return nullptr;
        }
        uint64_t size = numElements * sizeof(T);
        size = size + 16 - (size % 16);
        return reinterpret_cast<T*>(getSubArray(size));
    }

    __device__ __inline__ uint8_t* getRawSubArray(uint64_t size)
    {
        if (0 == size) {
            return nullptr;
        }
        size = size + 16 - (size % 16);
        return getSubArray(size);
    }

    template <typename T>
    __device__ __inline__ T& atType(uint64_t address)
    {
        return *reinterpret_cast<T*>(*_data + address);
    }
    template <typename T>
    __device__ __inline__ T const& atType(uint64_t address) const
    {
        return *reinterpret_cast<T*>(*_data + address);
    }
};

//data of UnmanagedArray is stored elsewhere
template <typename T>
class UnmanagedArray
{
private:
    int* _size;
    int* _numEntries;
    int* _numOrigEntries;

public:
    T** _data;

    UnmanagedArray() {}

    //methods for host
    __host__ __inline__ void init()
    {
        CudaMemoryManager::getInstance().acquireMemory<int>(1, _numEntries);
        CudaMemoryManager::getInstance().acquireMemory<int>(1, _numOrigEntries);
        CudaMemoryManager::getInstance().acquireMemory<int>(1, _size);
        CudaMemoryManager::getInstance().acquireMemory<T*>(1, _data);

        CHECK_FOR_CUDA_ERROR(cudaMemset(_numOrigEntries, 0, sizeof(int)));
        CHECK_FOR_CUDA_ERROR(cudaMemset(_numEntries, 0, sizeof(int)));
        CHECK_FOR_CUDA_ERROR(cudaMemset(_size, 0, sizeof(int)));
    }

    __host__ __inline__ void free()
    {
        CudaMemoryManager::getInstance().freeMemory(_numOrigEntries);
        CudaMemoryManager::getInstance().freeMemory(_numEntries);
        CudaMemoryManager::getInstance().freeMemory(_size);
        CudaMemoryManager::getInstance().freeMemory(_data);
    }

    //methods for device
    __device__ __inline__ void setMemory(T* data, int size) const
    {
        *_data = data;
        *_size = size;
        *_numEntries = 0;
        *_numOrigEntries = 0;
    }
    __device__ __inline__ int getSize() const { return *_size; }
    __device__ __inline__ int saveNumEntries() const { return *_numOrigEntries = *_numEntries; }
    __device__ __inline__ int getNumEntries() const { return *_numEntries; }
    __device__ __inline__ int getNumOrigEntries() const { return *_numOrigEntries; }

    __device__ __inline__ T& at(int index) { return (*_data)[index]; }
    __device__ __inline__ T const& at(int index) const { return (*_data)[index]; }

    //returns index if successful, otherwise -1
    __device__ __inline__ int tryAddEntry(T const& entry)   
    {
        auto index = atomicAdd(_numEntries, 1);
        if (index < *_size) {
            (*_data)[index] = entry;
            return index;
        } else {
            atomicSub(_numEntries, 1);
            return -1;
        }
    }

    __device__ __inline__ int tryGetEntries(int numEntries)
    {
        auto index = atomicAdd(_numEntries, numEntries);
        if (index < *_size) {
            return index;
        } else {
            atomicSub(_numEntries, numEntries);
            return -1;
        }
    }
};
