#include "RenderingData.cuh"

void RenderingData::init() {}

void RenderingData::resizeImageIfNecessary(int2 const& newSize)
{
    if (newSize.x * newSize.y > numPixels) {
        CudaMemoryManager::getInstance().freeMemory(imageData);
        CudaMemoryManager::getInstance().acquireMemory<uint64_t>(newSize.x * newSize.y, imageData);
        numPixels = newSize.x * newSize.y;
    }
}

void RenderingData::free()
{
    CudaMemoryManager::getInstance().freeMemory(imageData);
}
