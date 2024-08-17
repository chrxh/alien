#pragma once

#include "DensityMap.cuh"
#include "ActiveRadiationSources.cuh"

struct PreprocessedSimulationData
{
    DensityMap densityMap;
    ActiveRadiationSources activeRadiationSources;

    __host__ __inline__ void init(int2 const& worldSize)
    {
        densityMap.init(worldSize, 8);
        activeRadiationSources.init();
    }

    __host__ __inline__ void free()
    {
        densityMap.free();
        activeRadiationSources.free();
    }
};
