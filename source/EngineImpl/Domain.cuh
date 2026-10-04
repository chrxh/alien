#pragma once

#include <memory>

#include <EngineKernels/Definitions.cuh>

// Host-side handle of one domain of the domain decomposition
struct Domain
{
    int index = 0;
    int device = 0;
    std::shared_ptr<SimulationData> data;
    std::shared_ptr<SimulationStatistics> statistics;
};
