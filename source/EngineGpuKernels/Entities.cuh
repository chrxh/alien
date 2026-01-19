#pragma once

#include <EngineInterface/CudaSettings.h>

#include "Array.cuh"
#include "Base.cuh"
#include "Definitions.cuh"

struct Entities
{
    Array<Object*> objects;
    Array<Energy*> energies;

    Heap heap;

    void init();
    void free();

    __device__ void saveNumEntries();
};
