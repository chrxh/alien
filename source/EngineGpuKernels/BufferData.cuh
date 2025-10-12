#pragma once

#include <atomic>

#include "EngineInterface/RenderData.h"

#include "Base.cuh"
#include "Definitions.cuh"

struct BufferData
{
    cudaGraphicsResource* vertexBuffer = nullptr;
    cudaGraphicsResource* lineIndexBuffer = nullptr;

    void init();
    void registerBuffers(RenderBuffers const& buffers);
    void resizeObjectBufferIfNecessary(NumRenderObjects const& numRenderObjects, RenderBuffers& buffers);
    void free();
};
