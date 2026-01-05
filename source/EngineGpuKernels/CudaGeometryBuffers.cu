#include "CudaGeometryBuffers.cuh"

#if defined(_WIN32)
#include <windows.h>
#endif

#include <cuda_gl_interop.h>
#include <algorithm>
#include <vector>

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

    memoryManager.freeMemory(deviceCellBuffer);
    memoryManager.freeMemory(deviceEnergyParticleBuffer);
    memoryManager.freeMemory(deviceLocationBuffer);
    memoryManager.freeMemory(deviceSelectedObjectBuffer);
    memoryManager.freeMemory(deviceLineIndexBuffer);
    memoryManager.freeMemory(deviceTriangleIndexBuffer);
    memoryManager.freeMemory(deviceSelectedConnectionBuffer);
    memoryManager.freeMemory(deviceAttackEventBuffer);
    memoryManager.freeMemory(deviceDetonationEventBuffer);
}

void CudaGeometryBuffers::copyToOpenGL(GeometryBuffers const& geometryBuffers, NumRenderObjects const& numObjects)
{
    if (numObjects.cells > 0) {
        std::vector<CellVertexData> hostCellBuffer(numObjects.cells);
        CHECK_FOR_CUDA_ERROR(cudaMemcpy(hostCellBuffer.data(), deviceCellBuffer, numObjects.cells * sizeof(CellVertexData), cudaMemcpyDeviceToHost));
        geometryBuffers->setCellData(hostCellBuffer.data(), numObjects.cells);
    }

    if (numObjects.energyParticles > 0) {
        std::vector<EnergyParticleVertexData> hostEnergyParticleBuffer(numObjects.energyParticles);
        CHECK_FOR_CUDA_ERROR(cudaMemcpy(hostEnergyParticleBuffer.data(), deviceEnergyParticleBuffer, numObjects.energyParticles * sizeof(EnergyParticleVertexData), cudaMemcpyDeviceToHost));
        geometryBuffers->setEnergyParticleData(hostEnergyParticleBuffer.data(), numObjects.energyParticles);
    }

    if (numObjects.locations > 0) {
        std::vector<LocationVertexData> hostLocationBuffer(numObjects.locations);
        CHECK_FOR_CUDA_ERROR(cudaMemcpy(hostLocationBuffer.data(), deviceLocationBuffer, numObjects.locations * sizeof(LocationVertexData), cudaMemcpyDeviceToHost));
        geometryBuffers->setLocationData(hostLocationBuffer.data(), numObjects.locations);
    }

    if (numObjects.selectedObjects > 0) {
        std::vector<getObjectVertexData> hostSelectedObjectBuffer(numObjects.selectedObjects);
        CHECK_FOR_CUDA_ERROR(cudaMemcpy(hostSelectedObjectBuffer.data(), deviceSelectedObjectBuffer, numObjects.selectedObjects * sizeof(getObjectVertexData), cudaMemcpyDeviceToHost));
        geometryBuffers->setSelectedObjectData(hostSelectedObjectBuffer.data(), numObjects.selectedObjects);
    }

    if (numObjects.lineIndices > 0) {
        std::vector<unsigned int> hostLineIndexBuffer(numObjects.lineIndices);
        CHECK_FOR_CUDA_ERROR(cudaMemcpy(hostLineIndexBuffer.data(), deviceLineIndexBuffer, numObjects.lineIndices * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        geometryBuffers->setLineIndices(hostLineIndexBuffer.data(), numObjects.lineIndices);
    }

    if (numObjects.triangleIndices > 0) {
        std::vector<unsigned int> hostTriangleIndexBuffer(numObjects.triangleIndices);
        CHECK_FOR_CUDA_ERROR(cudaMemcpy(hostTriangleIndexBuffer.data(), deviceTriangleIndexBuffer, numObjects.triangleIndices * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        geometryBuffers->setTriangleIndices(hostTriangleIndexBuffer.data(), numObjects.triangleIndices);
    }

    if (numObjects.connectionArrowVertices > 0) {
        std::vector<ConnectionArrowVertexData> hostSelectedConnectionBuffer(numObjects.connectionArrowVertices);
        CHECK_FOR_CUDA_ERROR(cudaMemcpy(hostSelectedConnectionBuffer.data(), deviceSelectedConnectionBuffer, numObjects.connectionArrowVertices * sizeof(ConnectionArrowVertexData), cudaMemcpyDeviceToHost));
        geometryBuffers->setSelectedConnectionData(hostSelectedConnectionBuffer.data(), numObjects.connectionArrowVertices);
    }

    if (numObjects.attackEventVertices > 0) {
        std::vector<AttackEventVertexData> hostAttackEventBuffer(numObjects.attackEventVertices);
        CHECK_FOR_CUDA_ERROR(cudaMemcpy(hostAttackEventBuffer.data(), deviceAttackEventBuffer, numObjects.attackEventVertices * sizeof(AttackEventVertexData), cudaMemcpyDeviceToHost));
        geometryBuffers->setAttackEventData(hostAttackEventBuffer.data(), numObjects.attackEventVertices);
    }

    if (numObjects.detonationEventVertices > 0) {
        std::vector<DetonationEventVertexData> hostDetonationEventBuffer(numObjects.detonationEventVertices);
        CHECK_FOR_CUDA_ERROR(cudaMemcpy(hostDetonationEventBuffer.data(), deviceDetonationEventBuffer, numObjects.detonationEventVertices * sizeof(DetonationEventVertexData), cudaMemcpyDeviceToHost));
        geometryBuffers->setDetonationEventData(hostDetonationEventBuffer.data(), numObjects.detonationEventVertices);
    }
}
