#include "RenderingData.cuh"

#include "CudaMemoryManager.cuh"

void RenderingData::init()
{
    CudaMemoryManager::getInstance().acquireMemory<int>(1, numObjects);
}

void RenderingData::resizeImageIfNecessary(int2 const& newSize)
{
    if (newSize.x * newSize.y > numPixels) {
        CudaMemoryManager::getInstance().freeMemory(imageData);
        CudaMemoryManager::getInstance().acquireMemory<uint64_t>(newSize.x * newSize.y, imageData);
        numPixels = newSize.x * newSize.y;
    }
}

void RenderingData::resizeObjectBufferIfNecessary(int maxNumObjects)
{
    if (maxNumObjects > maxObjects) {
        CudaMemoryManager::getInstance().freeMemory(objectData);
        CudaMemoryManager::getInstance().acquireMemory<RenderingObjectData>(maxNumObjects, objectData);
        maxObjects = maxNumObjects;
    }
}

void RenderingData::free()
{
    CudaMemoryManager::getInstance().freeMemory(imageData);
    CudaMemoryManager::getInstance().freeMemory(objectData);
    CudaMemoryManager::getInstance().freeMemory(numObjects);
}
