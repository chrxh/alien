#include "CudaGeometryBuffers.cuh"
#include "CudaMemoryManager.cuh"

#if !defined(_WIN32)
#include <unistd.h>
#endif

#include <vector>

namespace
{
    cudaError_t importExternalMemory(cudaExternalMemory_t& result, SharedGeometryMemory const& memory)
    {
        cudaExternalMemoryHandleDesc description = {};
#if defined(_WIN32)
        description.type = cudaExternalMemoryHandleTypeOpaqueWin32;
        description.handle.win32.handle = memory.win32Handle;
#else
        description.type = cudaExternalMemoryHandleTypeOpaqueFd;
        description.handle.fd = memory.fd;
#endif
        description.size = memory.allocationSize;
        description.flags = memory.dedicatedAllocation ? cudaExternalMemoryDedicated : 0;

        auto importResult = cudaImportExternalMemory(&result, &description);

        // NT handles stay with the geometry buffers, file descriptors of successful imports belong to CUDA
#if !defined(_WIN32)
        if (importResult != cudaSuccess) {
            close(memory.fd);
        }
#endif
        return importResult;
    }

    cudaError_t mapExternalMemory(void*& result, cudaExternalMemory_t memory, uint64_t sizeInBytes)
    {
        cudaExternalMemoryBufferDesc description = {};
        description.offset = 0;
        description.size = sizeInBytes;
        return cudaExternalMemoryGetMappedBuffer(&result, memory, &description);
    }
}

cudaError_t CudaGeometryBuffers::importSharedMemory(GeometryBuffers const& geometryBuffers)
{
    if (_importedGeometryBuffers.lock() != geometryBuffers) {
        releaseSharedMemory();
        _importedGeometryBuffers = geometryBuffers;
    }

    // Only buffers with new memory are imported again
    for (GeometryBufferType type = 0; type < GeometryBufferType_Count; ++type) {
        auto allocationId = geometryBuffers->getAllocationId(type);
        if (_sharedBuffers.at(type) != nullptr && _importedAllocationIds.at(type) == allocationId) {
            continue;
        }
        releaseSharedMemory(type);
        auto sharedMemory = geometryBuffers->shareMemory(type);
        if (auto result = importExternalMemory(_externalMemories.at(type), sharedMemory); result != cudaSuccess) {
            releaseSharedMemory();
            return result;
        }
        auto sizeInBytes = geometryBuffers->getCapacity(type) * GeometryBufferLayout::ElementSizes.at(type);
        if (auto result = mapExternalMemory(_sharedBuffers.at(type), _externalMemories.at(type), sizeInBytes); result != cudaSuccess) {
            releaseSharedMemory();
            return result;
        }
        _importedAllocationIds.at(type) = allocationId;
    }
    _activeBuffers = _sharedBuffers;
    return cudaSuccess;
}

void CudaGeometryBuffers::allocateDeviceBuffers(GeometryBuffers const& geometryBuffers)
{
    auto& memoryManager = CudaMemoryManager::getInstance();
    for (GeometryBufferType type = 0; type < GeometryBufferType_Count; ++type) {
        auto sizeInBytes = geometryBuffers->getCapacity(type) * GeometryBufferLayout::ElementSizes.at(type);
        auto& deviceBuffer = _deviceBuffers.at(type);
        auto& deviceBufferSize = _deviceBufferSizes.at(type);
        if (sizeInBytes > deviceBufferSize) {
            memoryManager.freeMemory(deviceBuffer);
            memoryManager.acquireMemory(sizeInBytes, deviceBuffer);
            deviceBufferSize = sizeInBytes;
        }
        _activeBuffers.at(type) = deviceBuffer;
    }
}

void CudaGeometryBuffers::copyDeviceBuffersTo(GeometryBuffers const& geometryBuffers, NumRenderObjects const& numObjects)
{
    std::vector<uint8_t> hostBuffer;
    for (GeometryBufferType type = 0; type < GeometryBufferType_Count; ++type) {
        auto sizeInBytes = GeometryBufferLayout::getNumElements(numObjects, type) * GeometryBufferLayout::ElementSizes.at(type);
        if (sizeInBytes == 0) {
            continue;
        }
        hostBuffer.resize(sizeInBytes);
        CHECK_FOR_DEVICE_ERRORS(cudaMemcpy(hostBuffer.data(), _deviceBuffers.at(type), sizeInBytes, cudaMemcpyDeviceToHost));
        geometryBuffers->upload(type, hostBuffer.data(), sizeInBytes);
    }
}

void CudaGeometryBuffers::release()
{
    releaseSharedMemory();
    releaseDeviceBuffers();
}

void CudaGeometryBuffers::releaseSharedMemory()
{
    for (GeometryBufferType type = 0; type < GeometryBufferType_Count; ++type) {
        releaseSharedMemory(type);
    }
    _importedGeometryBuffers.reset();
    _activeBuffers = {};
}

void CudaGeometryBuffers::releaseSharedMemory(GeometryBufferType type)
{
    auto contextValid = !CudaContextState::get().isInvalid();
    auto& sharedBuffer = _sharedBuffers.at(type);
    auto& externalMemory = _externalMemories.at(type);
    if (contextValid && sharedBuffer != nullptr) {
        cudaFree(sharedBuffer);
    }
    if (contextValid && externalMemory != nullptr) {
        cudaDestroyExternalMemory(externalMemory);
    }
    sharedBuffer = nullptr;
    externalMemory = nullptr;
}

void CudaGeometryBuffers::releaseDeviceBuffers()
{
    auto& memoryManager = CudaMemoryManager::getInstance();
    for (GeometryBufferType type = 0; type < GeometryBufferType_Count; ++type) {
        memoryManager.freeMemory(_deviceBuffers.at(type));
        _deviceBufferSizes.at(type) = 0;
    }
    _activeBuffers = {};
}
