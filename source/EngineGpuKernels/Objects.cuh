#pragma once

#include <EngineInterface/CudaSettings.h>

#include "Array.cuh"
#include "Base.cuh"
#include "Definitions.cuh"

struct Objects
{
    Array<Cell*> cells;
    Array<Particle*> particles;

    Heap heap;

    void init();
    void free();

    __device__ void saveNumEntries();
};
