#include "CudaGeometryBuffers.cuh"

#if defined(_WIN32)
#include <windows.h>
#endif

#include <cuda_gl_interop.h>

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

void CudaGeometryBuffers::registerBuffers(GeometryBuffers const& buffers)
{
    if (vertexBuffer != nullptr) {
        unregisterBufferResource(vertexBuffer);
    }
    vertexBuffer = registerBufferResource(buffers->getVboForCells());

    if (energyParticleBuffer != nullptr) {
        unregisterBufferResource(energyParticleBuffer);
    }
    energyParticleBuffer = registerBufferResource(buffers->getVboForEnergyParticles());

    if (lineIndexBuffer != nullptr) {
        unregisterBufferResource(lineIndexBuffer);
    }
    lineIndexBuffer = registerBufferResource(buffers->getEboForLines());

    if (triangleIndexBuffer != nullptr) {
        unregisterBufferResource(triangleIndexBuffer);
    }
    triangleIndexBuffer = registerBufferResource(buffers->getEboForTriangles());
}
