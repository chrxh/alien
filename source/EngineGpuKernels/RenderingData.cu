#include "RenderingData.cuh"

#if defined(_WIN32)
#include <windows.h>
#endif

#include <cuda_gl_interop.h>


void RenderingData::init()
{
    CudaMemoryManager::getInstance().acquireMemory<uint64_t>(1, numVertices);
}

namespace
{
    cudaGraphicsResource* registerBufferResource(GLuint buffer)
    {
        cudaGraphicsResource* result = nullptr;
        CHECK_FOR_CUDA_ERROR(cudaGraphicsGLRegisterBuffer(&result, buffer, cudaGraphicsMapFlagsWriteDiscard));

        return result;
    }

    void unregisterBufferResource(cudaGraphicsResource* buffer)
    {
        CHECK_FOR_CUDA_ERROR(cudaGraphicsUnregisterResource(buffer));
    }
}

void RenderingData::registerBuffers(RenderBuffers const& buffers)
{
    if (vertexBuffer != nullptr) {
        unregisterBufferResource(vertexBuffer);
    }
    vertexBuffer = registerBufferResource(buffers.vboForPoints);
}

void RenderingData::resizeObjectBufferIfNecessary(uint64_t numRequiredObjects)
{
    //if (numRequiredObjects > capacity) {
    //    CudaMemoryManager::getInstance().freeMemory(vertices);
    //    CudaMemoryManager::getInstance().acquireMemory<VertexData>(numRequiredObjects * 2, vertices);
    //    capacity = numRequiredObjects * 2;
    //}
}

void RenderingData::free()
{
    CudaMemoryManager::getInstance().freeMemory(numVertices);
}
