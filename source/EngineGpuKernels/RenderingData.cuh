#pragma once

#include <atomic>

#include "Base.cuh"
#include "Definitions.cuh"

struct RenderingData
{
    int numPixels = 0;
    uint64_t* imageDataDevice = nullptr;  //pixel in bbbbggggrrrr format (3 x 16 bit + 16 bit unused)
    uint64_t* imageDataHost = nullptr;

    void init();
    void resizeImageIfNecessary(int2 const& newSize);
    void free();
};
