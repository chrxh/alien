#pragma once

#include "EngineInterface/GpuSettings.h"

#include "Base.cuh"
#include "Definitions.cuh"
#include "RawMemory.cuh"

struct Entities
{
    Array<Cell*> cellPointers;
    Array<Token*> tokenPointers;
    Array<Particle*> particlePointers;

    Array<Cell> cells;
    Array<Token> tokens;
    Array<Particle> particles;

    RawMemory stringBytes;

    void init();
    void free();

    __device__ __inline__ void saveNumEntries()
    {
        cellPointers.saveNumEntries();
        tokenPointers.saveNumEntries();
        particlePointers.saveNumEntries();
        cells.saveNumEntries();
        tokens.saveNumEntries();
        particles.saveNumEntries();
    }
};

