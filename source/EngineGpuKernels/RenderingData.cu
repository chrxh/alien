#include "RenderingData.cuh"

#if defined(_WIN32)
#include <windows.h>
#endif

#include <cuda_gl_interop.h>

#include "RenderingData.cuh"
#include "CudaMemoryManager.cuh"


void RenderingData::init()
{
    CudaMemoryManager::getInstance().acquireMemory<uint64_t>(1, numVertices);
}

namespace
{
    void* registerBufferResource(GLuint buffer)
    {
        cudaGraphicsResource* result = nullptr;
        CHECK_FOR_CUDA_ERROR(cudaGraphicsGLRegisterBuffer(&result, buffer, cudaGraphicsMapFlagsWriteDiscard));

        return reinterpret_cast<void*>(result);
    }

    void unregisterBufferResource(void* buffer)
    {
        CHECK_FOR_CUDA_ERROR(cudaGraphicsUnregisterResource(reinterpret_cast<cudaGraphicsResource*>(buffer)));
    }
}

void RenderingData::registerBuffers(RenderBuffers const& buffers)
{
    if (vertices_openGlBuffer != nullptr) {
        unregisterBufferResource(vertices_openGlBuffer);
    }
    vertices_openGlBuffer = registerBufferResource(buffers.vboForPoints);
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
        CudaMemoryManager::getInstance().freeMemory(vertices);
        CudaMemoryManager::getInstance().acquireMemory<VertexData>(numRequiredObjects * 2, vertices);
        capacity = numRequiredObjects * 2;
    }
}

void RenderingData::free()
{
    if (vertices_openGlBuffer != nullptr) {
        unregisterBufferResource(vertices_openGlBuffer);
    }

    CudaMemoryManager::getInstance().freeMemory(imageData);
    CudaMemoryManager::getInstance().freeMemory(vertices);
    CudaMemoryManager::getInstance().freeMemory(numVertices);
}
