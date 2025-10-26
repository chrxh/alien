#pragma once

#include "EngineInterface/GeometryBuffers.h"

#include "Base.cuh"
#include "Definitions.cuh"

struct CudaGeometryBuffers
{
    cudaGraphicsResource* vertexBuffer = nullptr;
    cudaGraphicsResource* energyParticleBuffer = nullptr;
    cudaGraphicsResource* locationBuffer = nullptr;
    cudaGraphicsResource* selectedObjectBuffer = nullptr;
    cudaGraphicsResource* lineIndexBuffer = nullptr;
    cudaGraphicsResource* triangleIndexBuffer = nullptr;
    cudaGraphicsResource* selectedConnectionBuffer = nullptr;

    void registerBuffers(GeometryBuffers const& buffers);
};
