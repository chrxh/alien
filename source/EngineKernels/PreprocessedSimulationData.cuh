#pragma once

#include "ActiveRadiationSources.cuh"
#include "DensityGrid.cuh"

struct PreprocessedSimulationData
{
    DensityGrid densityGrid;
    ActiveRadiationSources activeRadiationSources;

    __host__ __inline__ void init(int2 const& worldSize)
    {
        densityGrid.init(worldSize);
        activeRadiationSources.init();
    }

    __host__ __inline__ void free()
    {
        densityGrid.free();
        activeRadiationSources.free();
    }
};
