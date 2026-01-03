#include "CudaGeometryBuffers.cuh"

#if defined(_WIN32)
#include <windows.h>
#endif

#include <cuda_gl_interop.h>
#include <algorithm>

#include "CudaMemoryManager.cuh"

namespace
{
    cudaGraphicsResource* registerBufferResource(GLuint buffer)
    {
        cudaGraphicsResource* result = nullptr;
        CHECK_FOR_CUDA_ERROR(cudaGraphicsGLRegisterBuffer(&result, buffer, cudaGraphicsMapFlagsWriteDiscard));

        return result;
    }

    void unregisterBufferResource(cudaGraphicsResource* buffer)
    {
        CHECK_FOR_CUDA_ERROR(cudaGraphicsUnregisterResource(buffer));
    }
}

void CudaGeometryBuffers::registerBuffers(GeometryBuffers const& buffers)
{
    if (vertexBuffer != nullptr) {
        unregisterBufferResource(vertexBuffer);
    }
    vertexBuffer = registerBufferResource(buffers->getVboForCells());

    if (energyParticleBuffer != nullptr) {
        unregisterBufferResource(energyParticleBuffer);
    }
    energyParticleBuffer = registerBufferResource(buffers->getVboForEnergyParticles());

    if (locationBuffer != nullptr) {
        unregisterBufferResource(locationBuffer);
    }
    locationBuffer = registerBufferResource(buffers->getVboForLocations());

    if (selectedObjectBuffer != nullptr) {
        unregisterBufferResource(selectedObjectBuffer);
    }
    selectedObjectBuffer = registerBufferResource(buffers->getVboForSelectedObjects());

    if (lineIndexBuffer != nullptr) {
        unregisterBufferResource(lineIndexBuffer);
    }
    lineIndexBuffer = registerBufferResource(buffers->getEboForLines());

    if (triangleIndexBuffer != nullptr) {
        unregisterBufferResource(triangleIndexBuffer);
    }
    triangleIndexBuffer = registerBufferResource(buffers->getEboForTriangles());

    if (selectedConnectionBuffer != nullptr) {
        unregisterBufferResource(selectedConnectionBuffer);
    }
    selectedConnectionBuffer = registerBufferResource(buffers->getVboForSelectedConnections());

    if (attackEventBuffer != nullptr) {
        unregisterBufferResource(attackEventBuffer);
    }
    attackEventBuffer = registerBufferResource(buffers->getVboForAttackEvents());

    if (detonationEventBuffer != nullptr) {
        unregisterBufferResource(detonationEventBuffer);
    }
    detonationEventBuffer = registerBufferResource(buffers->getVboForDetonationEvents());
}

void CudaGeometryBuffers::allocateBuffersForNoInterop(NumRenderObjects const& numObjects)
{
    auto& memoryManager = CudaMemoryManager::getInstance();

    // Allocate or reallocate cell buffer
    auto requiredCellCapacity = std::max(numObjects.cells * 2, static_cast<uint64_t>(100000));
    if (numObjects.cells >= deviceCellBufferCapacity) {
        if (deviceCellBuffer != nullptr) {
            memoryManager.freeMemory(deviceCellBuffer);
        }
        memoryManager.acquireMemory(requiredCellCapacity, deviceCellBuffer);
        deviceCellBufferCapacity = requiredCellCapacity;
    }

    // Allocate or reallocate energy particle buffer
    auto requiredEnergyParticleCapacity = std::max(numObjects.energyParticles * 2, static_cast<uint64_t>(100000));
    if (numObjects.energyParticles >= deviceEnergyParticleBufferCapacity) {
        if (deviceEnergyParticleBuffer != nullptr) {
            memoryManager.freeMemory(deviceEnergyParticleBuffer);
        }
        memoryManager.acquireMemory(requiredEnergyParticleCapacity, deviceEnergyParticleBuffer);
        deviceEnergyParticleBufferCapacity = requiredEnergyParticleCapacity;
    }

    // Allocate or reallocate location buffer
    auto requiredLocationCapacity = std::max(numObjects.locations * 2, static_cast<uint64_t>(1000));
    if (numObjects.locations >= deviceLocationBufferCapacity) {
        if (deviceLocationBuffer != nullptr) {
            memoryManager.freeMemory(deviceLocationBuffer);
        }
        memoryManager.acquireMemory(requiredLocationCapacity, deviceLocationBuffer);
        deviceLocationBufferCapacity = requiredLocationCapacity;
    }

    // Allocate or reallocate selected object buffer
    auto requiredSelectedObjectCapacity = std::max(numObjects.selectedObjects * 2, static_cast<uint64_t>(10000));
    if (numObjects.selectedObjects >= deviceSelectedObjectBufferCapacity) {
        if (deviceSelectedObjectBuffer != nullptr) {
            memoryManager.freeMemory(deviceSelectedObjectBuffer);
        }
        memoryManager.acquireMemory(requiredSelectedObjectCapacity, deviceSelectedObjectBuffer);
        deviceSelectedObjectBufferCapacity = requiredSelectedObjectCapacity;
    }

    // Allocate or reallocate line index buffer
    auto requiredLineIndexCapacity = std::max(numObjects.lineIndices * 2, static_cast<uint64_t>(100000));
    if (numObjects.lineIndices >= deviceLineIndexBufferCapacity) {
        if (deviceLineIndexBuffer != nullptr) {
            memoryManager.freeMemory(deviceLineIndexBuffer);
        }
        memoryManager.acquireMemory(requiredLineIndexCapacity, deviceLineIndexBuffer);
        deviceLineIndexBufferCapacity = requiredLineIndexCapacity;
    }

    // Allocate or reallocate triangle index buffer
    auto requiredTriangleIndexCapacity = std::max(numObjects.triangleIndices * 2, static_cast<uint64_t>(100000));
    if (numObjects.triangleIndices >= deviceTriangleIndexBufferCapacity) {
        if (deviceTriangleIndexBuffer != nullptr) {
            memoryManager.freeMemory(deviceTriangleIndexBuffer);
        }
        memoryManager.acquireMemory(requiredTriangleIndexCapacity, deviceTriangleIndexBuffer);
        deviceTriangleIndexBufferCapacity = requiredTriangleIndexCapacity;
    }

    // Allocate or reallocate selected connection buffer
    auto requiredSelectedConnectionCapacity = std::max(numObjects.connectionArrowVertices * 2, static_cast<uint64_t>(100000));
    if (numObjects.connectionArrowVertices >= deviceSelectedConnectionBufferCapacity) {
        if (deviceSelectedConnectionBuffer != nullptr) {
            memoryManager.freeMemory(deviceSelectedConnectionBuffer);
        }
        memoryManager.acquireMemory(requiredSelectedConnectionCapacity, deviceSelectedConnectionBuffer);
        deviceSelectedConnectionBufferCapacity = requiredSelectedConnectionCapacity;
    }

    // Allocate or reallocate attack event buffer
    auto requiredAttackEventCapacity = std::max(numObjects.attackEventVertices * 2, static_cast<uint64_t>(10000));
    if (numObjects.attackEventVertices >= deviceAttackEventBufferCapacity) {
        if (deviceAttackEventBuffer != nullptr) {
            memoryManager.freeMemory(deviceAttackEventBuffer);
        }
        memoryManager.acquireMemory(requiredAttackEventCapacity, deviceAttackEventBuffer);
        deviceAttackEventBufferCapacity = requiredAttackEventCapacity;
    }

    // Allocate or reallocate detonation event buffer
    auto requiredDetonationEventCapacity = std::max(numObjects.detonationEventVertices * 2, static_cast<uint64_t>(10000));
    if (numObjects.detonationEventVertices >= deviceDetonationEventBufferCapacity) {
        if (deviceDetonationEventBuffer != nullptr) {
            memoryManager.freeMemory(deviceDetonationEventBuffer);
        }
        memoryManager.acquireMemory(requiredDetonationEventCapacity, deviceDetonationEventBuffer);
        deviceDetonationEventBufferCapacity = requiredDetonationEventCapacity;
    }
}

void CudaGeometryBuffers::freeBuffersForNoInterop()
{
    auto& memoryManager = CudaMemoryManager::getInstance();

    if (deviceCellBuffer != nullptr) {
        memoryManager.freeMemory(deviceCellBuffer);
        deviceCellBuffer = nullptr;
        deviceCellBufferCapacity = 0;
    }
    if (deviceEnergyParticleBuffer != nullptr) {
        memoryManager.freeMemory(deviceEnergyParticleBuffer);
        deviceEnergyParticleBuffer = nullptr;
        deviceEnergyParticleBufferCapacity = 0;
    }
    if (deviceLocationBuffer != nullptr) {
        memoryManager.freeMemory(deviceLocationBuffer);
        deviceLocationBuffer = nullptr;
        deviceLocationBufferCapacity = 0;
    }
    if (deviceSelectedObjectBuffer != nullptr) {
        memoryManager.freeMemory(deviceSelectedObjectBuffer);
        deviceSelectedObjectBuffer = nullptr;
        deviceSelectedObjectBufferCapacity = 0;
    }
    if (deviceLineIndexBuffer != nullptr) {
        memoryManager.freeMemory(deviceLineIndexBuffer);
        deviceLineIndexBuffer = nullptr;
        deviceLineIndexBufferCapacity = 0;
    }
    if (deviceTriangleIndexBuffer != nullptr) {
        memoryManager.freeMemory(deviceTriangleIndexBuffer);
        deviceTriangleIndexBuffer = nullptr;
        deviceTriangleIndexBufferCapacity = 0;
    }
    if (deviceSelectedConnectionBuffer != nullptr) {
        memoryManager.freeMemory(deviceSelectedConnectionBuffer);
        deviceSelectedConnectionBuffer = nullptr;
        deviceSelectedConnectionBufferCapacity = 0;
    }
    if (deviceAttackEventBuffer != nullptr) {
        memoryManager.freeMemory(deviceAttackEventBuffer);
        deviceAttackEventBuffer = nullptr;
        deviceAttackEventBufferCapacity = 0;
    }
    if (deviceDetonationEventBuffer != nullptr) {
        memoryManager.freeMemory(deviceDetonationEventBuffer);
        deviceDetonationEventBuffer = nullptr;
        deviceDetonationEventBufferCapacity = 0;
    }
}
