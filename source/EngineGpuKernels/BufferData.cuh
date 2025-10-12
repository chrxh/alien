#pragma once

#include <atomic>

#include "EngineInterface/GeometryBuffers.h"

#include "Base.cuh"
#include "Definitions.cuh"

struct BufferData
{
    cudaGraphicsResource* vertexBuffer = nullptr;
    cudaGraphicsResource* lineIndexBuffer = nullptr;

    void init();
    void registerBuffers(GeometryBuffers const& buffers);
    void resizeObjectBufferIfNecessary(NumRenderObjects const& numRenderObjects, GeometryBuffers const& buffers);
    void free();
};
