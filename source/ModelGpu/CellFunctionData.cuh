#pragma once

#include "ClustersByMapSection.cuh"

struct CellFunctionData
{
    ClustersByMapSection clustersByMapSection;

    void init(int2 const& universeSize)
    {
        clustersByMapSection.init(universeSize, { 50,50 });
    }

    void free()
    {
        clustersByMapSection.free();
    }
};
