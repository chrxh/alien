#pragma once

#include <helper_cuda.h>

#include "Base.cuh"
#include "Macros.cuh"

class CudaMemoryManager
{
public:
    static CudaMemoryManager& getInstance()
    {
        static CudaMemoryManager instance;
        return instance;
    }

    CudaMemoryManager(CudaMemoryManager const&) = delete;
    void operator= (CudaMemoryManager const&) = delete;

    void reset()
    {
        _bytes = 0;
    }

    template<typename T>
    void acquireMemory(uint64_t arraySize, T*& result)
    {
        CHECK_FOR_CUDA_ERROR(cudaMalloc(&result, sizeof(T)*arraySize));
        _bytes += sizeof(T)*arraySize;
    }

    template<typename T>
    void freeMemory(uint64_t arraySize, T& memory)
    {
        CHECK_FOR_CUDA_ERROR(cudaFree(memory));
        _bytes -= sizeof(T) * arraySize;
    }

    uint64_t getSizeOfAcquiredMemory() const
    {
        return _bytes;
    }

private:
    CudaMemoryManager() {}
    ~CudaMemoryManager() {}

    uint64_t _bytes = 0;
};