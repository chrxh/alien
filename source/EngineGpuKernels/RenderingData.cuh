#pragma once

#include <atomic>

#include <stb_image.h>

#include "Base.cuh"
#include "Textures.cuh"
#include "Definitions.cuh"

struct RenderingData
{
    int numPixels = 0;
    uint64_t* imageData = nullptr;  //pixel in bbbbggggrrrr format (3 x 16 bit + 16 bit unused)

    void init()
    {
    }

    void resizeImageIfNecessary(int2 const& newSize)
    {
        if (newSize.x * newSize.y > numPixels) {
            CudaMemoryManager::getInstance().freeMemory(imageData);
            CudaMemoryManager::getInstance().acquireMemory<uint64_t>(newSize.x * newSize.y, imageData);
            numPixels = newSize.x * newSize.y;
        }
    }

    void free()
    {
        CudaMemoryManager::getInstance().freeMemory(imageData);
        CudaMemoryManager::getInstance().freeMemory(textures.computingFunction.data);
    }
};
