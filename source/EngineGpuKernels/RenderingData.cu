#include "RenderingData.cuh"

void RenderingData::init() {}

void RenderingData::resizeImageIfNecessary(int2 const& newSize)
{
    if (newSize.x * newSize.y > numPixels) {
        CudaMemoryManager::getInstance().freeMemory(imageDataDevice);
        CudaMemoryManager::getInstance().acquireMemory<uint64_t>(newSize.x * newSize.y, imageDataDevice);
        delete[] imageDataHost;
        imageDataHost = new uint64_t[newSize.x * newSize.y];
        numPixels = newSize.x * newSize.y;
    }
}

void RenderingData::free()
{
    CudaMemoryManager::getInstance().freeMemory(imageDataDevice);
    delete[] imageDataHost;
}
