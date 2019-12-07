#pragma once

#include "MapSectionCollector.cuh"

struct CellFunctionData
{
    MapSectionCollector mapSectionCollector;

    __host__ __inline__ void init(int2 const& universeSize)
    {
        mapSectionCollector.init(universeSize, { 50,50 });
    }

    __host__ __inline__ void free()
    {
        mapSectionCollector.free();
    }
};
