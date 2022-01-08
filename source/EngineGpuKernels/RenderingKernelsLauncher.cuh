#pragma once

#include "EngineInterface/GpuSettings.h"
#include "EngineInterface/ShallowUpdateSelectionData.h"

#include "Base.cuh"
#include "DataAccessKernels.cuh"
#include "Definitions.cuh"
#include "GarbageCollectorKernelsLauncher.cuh"
#include "Macros.cuh"

class RenderingKernelsLauncher
{
public:
    void drawImage(
        GpuSettings const& gpuSettings,
        float2 rectUpperLeft,
        float2 rectLowerRight,
        int2 imageSize,
        float zoom,
        SimulationData data,
        RenderingData renderingData);
};
