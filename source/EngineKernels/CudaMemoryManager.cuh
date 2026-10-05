#pragma once

#include <map>

#include <cuda/helper_cuda.h>
#include "Macros.cuh"

inline int getCurrentDevice()
{
    int result = 0;
    CHECK_FOR_DEVICE_ERRORS(cudaGetDevice(&result));
    return result;
}

// Makes the given device the current one for the lifetime of the scope
class DeviceScope
{
public:
    explicit DeviceScope(int device)
        : _previousDevice(getCurrentDevice())
        , _switched(device != _previousDevice)
    {
        if (_switched) {
            CHECK_FOR_DEVICE_ERRORS(cudaSetDevice(device));
        }
    }
    ~DeviceScope()
    {
        if (_switched) {
            cudaSetDevice(_previousDevice);
        }
    }

    DeviceScope(DeviceScope const&) = delete;
    void operator=(DeviceScope const&) = delete;

private:
    int _previousDevice;
    bool _switched;
};

class CudaMemoryManager
{
public:
    static CudaMemoryManager& getInstance()
    {
        static CudaMemoryManager instance;
        return instance;
    }

    CudaMemoryManager(CudaMemoryManager const&) = delete;
    void operator=(CudaMemoryManager const&) = delete;

    void reset()
    {
        _bytes = 0;
        _allocations.clear();
    }

    template <typename T>
    void acquireMemory(uint64_t arraySize, T*& result)
    {
        CHECK_FOR_DEVICE_ERRORS(cudaMalloc(&result, sizeof(T) * arraySize));
        _bytes += sizeof(T) * arraySize;
        _allocations.emplace(reinterpret_cast<void*>(result), Allocation{sizeof(T) * arraySize, getCurrentDevice()});
    }

    // The memory may belong to another device than the current one
    template <typename T>
    void freeMemory(T*& memory)
    {
        if (!memory) {
            return;
        }
        auto const pointer = reinterpret_cast<void*>(memory);
        auto findResult = _allocations.find(pointer);
        if (findResult != _allocations.end()) {
            if (!CudaContextState::get().isInvalid()) {
                DeviceScope deviceScope(findResult->second.device);
                CHECK_FOR_DEVICE_ERRORS(cudaFree(memory));
            }
            _bytes -= findResult->second.numBytes;
            _allocations.erase(findResult);
        }
        memory = nullptr;
    }

    uint64_t getSizeOfAcquiredMemory() const { return _bytes; }

private:
    CudaMemoryManager() {}
    ~CudaMemoryManager() {}

    struct Allocation
    {
        uint64_t numBytes;
        int device;
    };

    uint64_t _bytes = 0;
    std::map<void*, Allocation> _allocations;
};
