#pragma once

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

    void init(GpuConstants const& gpuConstants)
    {
        cellPointers.init();
        cells.init();
        tokenPointers.init();
        tokens.init();
        particles.init();
        particlePointers.init();
/*
        cellPointers.init(gpuConstants.MAX_CELLPOINTERS);
        cells.init(gpuConstants.MAX_CELLS);
        tokenPointers.init(gpuConstants.MAX_TOKENPOINTERS);
        tokens.init(gpuConstants.MAX_TOKENS);
        particles.init(gpuConstants.MAX_PARTICLES);
        particlePointers.init(gpuConstants.MAX_PARTICLEPOINTERS);
*/
        strings.init();
        strings.resize(gpuConstants.METADATA_DYNAMIC_MEMORY_SIZE);
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

