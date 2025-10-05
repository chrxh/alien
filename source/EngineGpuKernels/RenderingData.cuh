#pragma once

#include <atomic>

#include "EngineInterface/RenderData.h"

#include "Base.cuh"
#include "Definitions.cuh"

struct RenderingData
{
    int numPixels = 0;
    uint64_t* imageData = nullptr;  //pixel in bbbbggggrrrr format (3 x 16 bit + 16 bit unused)

    // New shader-based rendering data
    uint64_t capacity = 0;
    uint64_t* numVertices = nullptr;
    VertexData* vertices = nullptr;
    void* vertices_openGlBuffer = nullptr;

    void init();
    void registerBuffers(RenderBuffers const& buffers);
    void resizeImageIfNecessary(int2 const& newSize);
    void resizeObjectBufferIfNecessary(uint64_t numRequiredObjects);
    void free();
};
