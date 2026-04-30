#pragma once

#include <EngineInterface/GeometryBuffers.h>

#include "Base.cuh"

struct CudaGeometryBuffers
{
    // CUDA-OpenGL interop resources (used when interop is enabled)
    cudaGraphicsResource* vertexBuffer = nullptr;
    cudaGraphicsResource* fluidParticleBuffer = nullptr;
    cudaGraphicsResource* locationBuffer = nullptr;
    cudaGraphicsResource* selectedObjectBuffer = nullptr;
    cudaGraphicsResource* lineIndexBuffer = nullptr;
    cudaGraphicsResource* triangleIndexBuffer = nullptr;
    cudaGraphicsResource* selectedConnectionBuffer = nullptr;
    cudaGraphicsResource* attackEventBuffer = nullptr;
    cudaGraphicsResource* detonationEventBuffer = nullptr;

    // CUDA device buffers for non-interop mode (data is copied to CPU then uploaded to OpenGL)
    ObjectVertexData* deviceObjectBuffer = nullptr;
    FluidParticleVertexData* deviceFluidParticleBuffer = nullptr;
    LocationVertexData* deviceLocationBuffer = nullptr;
    SelectedObjectVertexData* deviceSelectedObjectBuffer = nullptr;
    unsigned int* deviceLineIndexBuffer = nullptr;
    unsigned int* deviceTriangleIndexBuffer = nullptr;
    ConnectionArrowVertexData* deviceSelectedConnectionBuffer = nullptr;
    AttackEventVertexData* deviceAttackEventBuffer = nullptr;
    DetonationEventVertexData* deviceDetonationEventBuffer = nullptr;

    // Capacity tracking for device buffers
    uint64_t deviceObjectBufferCapacity = 0;
    uint64_t deviceFluidParticleBufferCapacity = 0;
    uint64_t deviceLocationBufferCapacity = 0;
    uint64_t deviceSelectedObjectBufferCapacity = 0;
    uint64_t deviceLineIndexBufferCapacity = 0;
    uint64_t deviceTriangleIndexBufferCapacity = 0;
    uint64_t deviceSelectedConnectionBufferCapacity = 0;
    uint64_t deviceAttackEventBufferCapacity = 0;
    uint64_t deviceDetonationEventBufferCapacity = 0;

    void registerBuffers(GeometryBuffers const& buffers);
    void allocateBuffersForNoInterop(NumRenderObjects const& numObjects);
    void freeBuffersForNoInterop();
    void copyToOpenGL(GeometryBuffers const& geometryBuffers, NumRenderObjects const& numObjects);

private:
    // Cached identity of the last successfully-registered GeometryBuffers source.
    // registerBuffers() is a no-op when the source pointer, the per-buffer GL
    // object IDs, and the buffer storage version have not changed since the
    // previous call. Storage reallocations (glBufferData) bump the version so
    // CUDA resources are refreshed.
    _GeometryBuffers const* _registeredSource = nullptr;
    uint64_t _registeredVersion = 0;
    unsigned int _registeredVboObjects = 0;
    unsigned int _registeredVboFluidParticles = 0;
    unsigned int _registeredVboLocations = 0;
    unsigned int _registeredVboSelectedObjects = 0;
    unsigned int _registeredEboLines = 0;
    unsigned int _registeredEboTriangles = 0;
    unsigned int _registeredVboSelectedConnections = 0;
    unsigned int _registeredVboAttackEvents = 0;
    unsigned int _registeredVboDetonationEvents = 0;
};
