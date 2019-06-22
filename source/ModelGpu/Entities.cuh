#pragma once

#include "Base.cuh"
#include "Definitions.cuh"

struct Entities
{
    ArrayController<Cluster*> clusterPointers;
    ArrayController<Cell*> cellPointers;
    ArrayController<Token*> tokenPointers;

    ArrayController<Cluster> clusters;
    ArrayController<Cell> cells;
    ArrayController<Token> tokens;
    ArrayController<Particle> particles;

    void init()
    {
        clusterPointers.init(MAX_CELLCLUSTERPOINTERS);
        clusters.init(MAX_CELLCLUSTERS);
        cellPointers.init(MAX_CELLPOINTERS);
        cells.init(MAX_CELLS);
        tokenPointers.init(MAX_TOKENPOINTERS);
        tokens.init(MAX_TOKENS);
        particles.init(MAX_PARTICLES);
    }

    void free()
    {
        clusterPointers.free();
        clusters.free();
        cellPointers.free();
        cells.free();
        tokenPointers.free();
        tokens.free();
        particles.free();
    }
};

