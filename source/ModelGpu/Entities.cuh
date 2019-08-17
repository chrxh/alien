#pragma once

#include "Base.cuh"
#include "Definitions.cuh"
#include "ClusterPointerMultiArrayController.cuh"

struct Entities
{
    ClusterPointerMultiArrayController<Cluster*> clusterPointerArrays;
    Array<Cell*> cellPointers;
    Array<Token*> tokenPointers;
    Array<Particle*> particlePointers;

    Array<Cluster> clusters;
    Array<Cell> cells;
    Array<Token> tokens;
    Array<Particle> particles;

    void init(CudaConstants const& cudaConstants)
    {
        int* sizes = new int[cudaConstants.NUM_CLUSTERPOINTERARRAYS];
        for (int i = 0; i < cudaConstants.NUM_CLUSTERPOINTERARRAYS; ++i) {
            sizes[i] = cudaConstants.MAX_CLUSTERPOINTERS / cudaConstants.NUM_CLUSTERPOINTERARRAYS;
        }
        clusterPointerArrays.init(cudaConstants.NUM_CLUSTERPOINTERARRAYS, sizes);
        delete[] sizes;

        clusters.init(cudaConstants.MAX_CLUSTERS);
        cellPointers.init(cudaConstants.MAX_CELLPOINTERS);
        cells.init(cudaConstants.MAX_CELLS);
        tokenPointers.init(cudaConstants.MAX_TOKENPOINTERS);
        tokens.init(cudaConstants.MAX_TOKENS);
        particles.init(cudaConstants.MAX_PARTICLES);
        particlePointers.init(cudaConstants.MAX_PARTICLEPOINTERS);
    }

    void free()
    {
        clusterPointerArrays.free();
        clusters.free();
        cellPointers.free();
        cells.free();
        tokenPointers.free();
        tokens.free();
        particles.free();
        particlePointers.free();
    }
};

