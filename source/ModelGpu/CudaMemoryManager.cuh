#pragma once

#include <helper_cuda.h>

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
    void acquireMemory(int64_t arraySize, T*& result)
    {
        checkCudaErrors(cudaMalloc(&result, sizeof(T)*arraySize));
        _bytes += sizeof(T)*arraySize;
    }

    template<typename T>
    void freeMemory(T& memory)
    {
        checkCudaErrors(cudaFree(memory));
    }

    int64_t getSizeOfAcquiredMemory() const
    {
        return _bytes;
    }

private:
    CudaMemoryManager() {}
    ~CudaMemoryManager() {}

    int64_t _bytes = 0;
};