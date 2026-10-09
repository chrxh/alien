#pragma once

#include <array>
#include <memory>

#include <EngineInterface/GeometryBuffers.h>

#include "Base.cuh"

// Device memory the geometry kernels write the rendering data into
struct CudaGeometryBuffers
{
public:
    // Interop mode: the kernels write directly into the memory of the geometry buffers
    cudaError_t importSharedMemory(GeometryBuffers const& geometryBuffers);

    // Host transfer mode: the kernels write into own device buffers, which are then copied to the geometry buffers over host memory
    void allocateDeviceBuffers(GeometryBuffers const& geometryBuffers);
    void copyDeviceBuffersTo(GeometryBuffers const& geometryBuffers, NumRenderObjects const& numObjects);

    void release();

    template <typename T>
    T* getBuffer(GeometryBufferType type) const
    {
        return static_cast<T*>(_activeBuffers.at(type));
    }

private:
    void releaseSharedMemory();
    void releaseSharedMemory(GeometryBufferType type);
    void releaseDeviceBuffers();

    std::array<void*, GeometryBufferType_Count> _activeBuffers = {};

    std::weak_ptr<_GeometryBuffers> _importedGeometryBuffers;
    std::array<uint64_t, GeometryBufferType_Count> _importedAllocationIds = {};
    std::array<cudaExternalMemory_t, GeometryBufferType_Count> _externalMemories = {};
    std::array<void*, GeometryBufferType_Count> _sharedBuffers = {};

    std::array<uint8_t*, GeometryBufferType_Count> _deviceBuffers = {};
    std::array<uint64_t, GeometryBufferType_Count> _deviceBufferSizes = {};
};
