#pragma once

#include "EngineInterface/GpuSettings.h"

#include "Base.cuh"
#include "Definitions.cuh"
#include "Array.cuh"

struct Objects
{
    Array<Cell*> cells;
    Array<Particle*> particles;

    Heap heap;

    void init();
    void free();

    __device__ void saveNumEntries();
};

