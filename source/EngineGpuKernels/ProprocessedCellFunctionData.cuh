#pragma once

#include "DensityMap.cuh"

struct PreprocessedCellFunctionData
{
    DensityMap densityMap;

    __host__ __inline__ void init(int2 const& worldSize)
    {
        densityMap.init(worldSize, 8);
    }

    __host__ __inline__ void free() { densityMap.free(); }
};
