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
    Textures textures;

    void init()
    {
        //#TODO load textures for cell function
/*
        int nrChannels;
        auto data = stbi_load(
            "Resources/function computer.png",
            &textures.computingFunction.width,
            &textures.computingFunction.height,
            &nrChannels,
            0);
        if (!data) {
            throw std::runtime_error("Failed to load texture");
        }
        uint64_t numBytes = textures.computingFunction.width * textures.computingFunction.height * 4;
        CudaMemoryManager::getInstance().acquireMemory<unsigned char>(numBytes, textures.computingFunction.data);
        CHECK_FOR_CUDA_ERROR(cudaMemcpy(textures.computingFunction.data, data, numBytes, cudaMemcpyHostToDevice));

        delete data;
*/
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
