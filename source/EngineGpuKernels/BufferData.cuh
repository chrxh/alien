#pragma once

#include <atomic>

#include "EngineInterface/RenderData.h"

#include "Base.cuh"
#include "Definitions.cuh"

struct BufferData
{
    uint64_t vertexBufferCapacity = 0;
    uint64_t LineIndexBufferCapacity = 0;
    cudaGraphicsResource* vertexBuffer = nullptr;
    cudaGraphicsResource* lineIndexBuffer = nullptr;

    void init();
    void registerBuffers(RenderBuffers const& buffers);
    void resizeObjectBufferIfNecessary(NumRenderObjects const& numRenderObjects, RenderBuffers const& buffers);
    void free();
};
