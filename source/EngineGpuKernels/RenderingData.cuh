#pragma once

#include <atomic>

#include "EngineInterface/RenderData.h"

#include "Base.cuh"
#include "Definitions.cuh"

struct RenderingData
{
    uint64_t capacity = 0;
    uint64_t* numVertices = nullptr;
    cudaGraphicsResource* vertexBuffer = nullptr;

    void init();
    void registerBuffers(RenderBuffers const& buffers);
    void resizeObjectBufferIfNecessary(uint64_t numRequiredObjects);
    void free();
};
