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
};
