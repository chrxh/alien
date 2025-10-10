#include "RenderingData.cuh"

#if defined(_WIN32)
#include <windows.h>
#endif

#include <GL/gl.h>
#include <cuda_gl_interop.h>

// OpenGL extension functions - declare as extern
extern "C" {
    void glBindBuffer(GLenum target, GLuint buffer);
    void glBufferData(GLenum target, GLsizeiptr size, const void* data, GLenum usage);
}


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


void RenderingData::resizeObjectBufferIfNecessary(NumRenderObjects const& numRenderObjects)
{
    if (numRenderObjects.vertices > capacity) {
        capacity = numRenderObjects.vertices * 2;
        glBindBuffer(GL_ARRAY_BUFFER, reinterpret_cast<uintptr_t>(vertexBuffer));
        glBufferData(GL_ARRAY_BUFFER, capacity * sizeof(VertexData), nullptr, GL_DYNAMIC_DRAW);
    }
}

void RenderingData::free()
{
    CudaMemoryManager::getInstance().freeMemory(numVertices);
}
