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

    void init(CudaConstants const& cudaConstants)
    {
        cellPointers.init(cudaConstants.MAX_CELLPOINTERS);
        cells.init(cudaConstants.MAX_CELLS);
        tokenPointers.init(cudaConstants.MAX_TOKENPOINTERS);
        tokens.init(cudaConstants.MAX_TOKENS);
        particles.init(cudaConstants.MAX_PARTICLES);
        particlePointers.init(cudaConstants.MAX_PARTICLEPOINTERS);
        strings.init(cudaConstants.METADATA_DYNAMIC_MEMORY_SIZE);
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

