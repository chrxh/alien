#include "BufferData.cuh"

#if defined(_WIN32)
#include <windows.h>
#endif

#include <glad/glad.h>
#include <cuda_gl_interop.h>

#include "CudaMemoryManager.cuh"


void BufferData::init()
{
    CudaMemoryManager::getInstance().acquireMemory<uint64_t>(1, numLineIndices);
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

void BufferData::registerBuffers(RenderBuffers const& buffers)
{
    if (vertexBuffer != nullptr) {
        unregisterBufferResource(vertexBuffer);
    }
    vertexBuffer = registerBufferResource(buffers.vboForPoints);

    //if (lineIndexBuffer != nullptr) {
    //    unregisterBufferResource(lineIndexBuffer);
    //}
    //lineIndexBuffer = registerBufferResource(buffers.eboForLines);
}


void BufferData::resizeObjectBufferIfNecessary(NumRenderObjects const& numRenderObjects, RenderBuffers const& buffers)
{
    if (numRenderObjects.vertices >= capacity) {
        capacity = max(numRenderObjects.vertices * 2, static_cast<uint64_t>(100000));
        glBindBuffer(GL_ARRAY_BUFFER, buffers.vboForPoints);
        glBufferData(GL_ARRAY_BUFFER, toInt(capacity * sizeof(VertexData)), nullptr, GL_DYNAMIC_DRAW);
    }
    //if (numRenderObjects.lineIndices >= lineIndexCapacity) {
    //    lineIndexCapacity = max(numRenderObjects.lineIndices * 2, static_cast<uint64_t>(100000));
    //    glBindVertexArray(buffers.vaoForLines);
    //    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffers.eboForLines);
    //    glBufferData(GL_ELEMENT_ARRAY_BUFFER, toInt(lineIndexCapacity * sizeof(unsigned int)), nullptr, GL_DYNAMIC_DRAW);
    //}
}

void BufferData::free()
{
    CudaMemoryManager::getInstance().freeMemory(numLineIndices);
}
