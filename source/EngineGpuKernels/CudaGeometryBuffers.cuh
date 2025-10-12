#pragma once

#include "EngineInterface/GeometryBuffers.h"

#include "Base.cuh"
#include "Definitions.cuh"

struct CudaGeometryBuffers
{
    cudaGraphicsResource* vertexBuffer = nullptr;
    cudaGraphicsResource* lineIndexBuffer = nullptr;

    void registerBuffers(GeometryBuffers const& buffers);
};
