#pragma once

#include <map>

#include <cuda/helper_cuda.h>

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
        _pointerToSizeMap.emplace(reinterpret_cast<void*>(result), arraySize);
    }

    template<typename T>
    void freeMemory(T*& memory)
    {
        if (!memory) {
            return;
        }
        auto findResult = _pointerToSizeMap.find(reinterpret_cast<void*>(memory));
        if (findResult != _pointerToSizeMap.end()) {
            CHECK_FOR_CUDA_ERROR(cudaFree(memory));
            _bytes -= sizeof(T) * findResult->second;
            _pointerToSizeMap.erase(findResult->first);
        }
    }

    uint64_t getSizeOfAcquiredMemory() const
    {
        return _bytes;
    }

private:
    CudaMemoryManager() {}
    ~CudaMemoryManager() {}

    uint64_t _bytes = 0;
    std::map<void*, uint64_t> _pointerToSizeMap;
};