#include "BufferData.cuh"

#if defined(_WIN32)
#include <windows.h>
#endif

#include <glad/glad.h>
#include <cuda_gl_interop.h>

#include "CudaMemoryManager.cuh"


void BufferData::init()
{
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

void BufferData::registerBuffers(GeometryBuffers const& buffers)
{
    if (vertexBuffer != nullptr) {
        unregisterBufferResource(vertexBuffer);
    }
    vertexBuffer = registerBufferResource(buffers->getVbo());

    if (lineIndexBuffer != nullptr) {
        unregisterBufferResource(lineIndexBuffer);
    }
    lineIndexBuffer = registerBufferResource(buffers->getEbo());
}


void BufferData::resizeObjectBufferIfNecessary(NumRenderObjects const& numRenderObjects, GeometryBuffers const& buffers)
{
    if (numRenderObjects.vertices >= buffers->vertexBufferCapacity) {
        buffers->vertexBufferCapacity = max(numRenderObjects.vertices * 2, static_cast<uint64_t>(100000));
        glBindBuffer(GL_ARRAY_BUFFER, buffers->getVbo());
        glBufferData(GL_ARRAY_BUFFER, toInt(buffers->vertexBufferCapacity * sizeof(VertexData)), nullptr, GL_DYNAMIC_DRAW);
    }
    if (numRenderObjects.lineIndices >= buffers->lineIndexBufferCapacity) {
        buffers->lineIndexBufferCapacity = max(numRenderObjects.lineIndices * 2, static_cast<uint64_t>(100000));
        glBindVertexArray(buffers->getVao());
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffers->getEbo());
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, toInt(buffers->lineIndexBufferCapacity * sizeof(unsigned int)), nullptr, GL_DYNAMIC_DRAW);
    }
}

void BufferData::free()
{
}
