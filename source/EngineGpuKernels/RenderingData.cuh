#pragma once

#include <atomic>

#include "Base.cuh"
#include "Definitions.cuh"

struct RenderingObjectData
{
    float2 pos;
    float3 color;
    float radius;
};

struct RenderingData
{
    int numPixels = 0;
    uint64_t* imageData = nullptr;  //pixel in bbbbggggrrrr format (3 x 16 bit + 16 bit unused)

    // New shader-based rendering data
    int maxObjects = 0;
    RenderingObjectData* objectData = nullptr;
    int* numObjects = nullptr;

    void init();
    void resizeImageIfNecessary(int2 const& newSize);
    void resizeObjectBufferIfNecessary(int maxNumObjects);
    void free();
};
