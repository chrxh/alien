#include "RenderingData.cuh"

#include "CudaMemoryManager.cuh"

void RenderingData::init()
{
    CudaMemoryManager::getInstance().acquireMemory<uint64_t>(1, numObjects);
}

void RenderingData::resizeImageIfNecessary(int2 const& newSize)
{
    if (newSize.x * newSize.y > numPixels) {
        CudaMemoryManager::getInstance().freeMemory(imageData);
        CudaMemoryManager::getInstance().acquireMemory<uint64_t>(newSize.x * newSize.y, imageData);
        numPixels = newSize.x * newSize.y;
    }
}

void RenderingData::resizeObjectBufferIfNecessary(uint64_t numRequiredObjects)
{
    if (numRequiredObjects > capacity) {
        CudaMemoryManager::getInstance().freeMemory(objectData);
        CudaMemoryManager::getInstance().acquireMemory<ObjectRenderData>(numRequiredObjects * 2, objectData);
        capacity = numRequiredObjects * 2;
    }
}

void RenderingData::free()
{
    CudaMemoryManager::getInstance().freeMemory(imageData);
    CudaMemoryManager::getInstance().freeMemory(objectData);
    CudaMemoryManager::getInstance().freeMemory(numObjects);
}
