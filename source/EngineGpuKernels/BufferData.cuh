#pragma once

#include <atomic>

#include "EngineInterface/RenderData.h"

#include "Base.cuh"
#include "Definitions.cuh"

struct BufferData
{
    uint64_t capacity = 0;
    uint64_t lineIndexCapacity = 0;
    uint64_t* numLineIndices = nullptr;
    cudaGraphicsResource* vertexBuffer = nullptr;
    cudaGraphicsResource* lineIndexBuffer = nullptr;

    void init();
    void registerBuffers(RenderBuffers const& buffers);
    void resizeObjectBufferIfNecessary(NumRenderObjects const& numRenderObjects, RenderBuffers const& buffers);
    void free();
};
