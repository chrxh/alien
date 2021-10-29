#pragma once

#include "EngineInterface/GpuSettings.h"

#include "Base.cuh"
#include "Definitions.cuh"
#include "DynamicMemory.cuh"

struct Entities
{
    Array<Cell*> cellPointers;
    Array<Token*> tokenPointers;
    Array<Particle*> particlePointers;

    Array<Cell> cells;
    Array<Token> tokens;
    Array<Particle> particles;

    DynamicMemory strings;

    void init()
    {
        cellPointers.init();
        cells.init();
        tokenPointers.init();
        tokens.init();
        particles.init();
        particlePointers.init();
        strings.init();
        strings.resize(Const::MetadataMemorySize);
    }

    void free()
    {
        cellPointers.free();
        cells.free();
        tokenPointers.free();
        tokens.free();
        particles.free();
        particlePointers.free();
        strings.free();
    }
};

