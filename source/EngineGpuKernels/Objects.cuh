#pragma once

#include "EngineInterface/GpuSettings.h"

#include "Base.cuh"
#include "Definitions.cuh"
#include "Array.cuh"

struct Objects
{
    Array<Cell*> cellPointers;
    Array<Particle*> particlePointers;

    Array<Cell> cells;
    Array<Particle> particles;

    RawMemory auxiliaryData;

    void init();
    void free();

    __device__ void saveNumEntries();
};

