#pragma once

#include "EngineInterface/GpuSettings.h"
#include "EngineInterface/ShallowUpdateSelectionData.h"
#include "EngineInterface/Settings.h"

#include "Base.cuh"
#include "DataAccessKernels.cuh"
#include "Definitions.cuh"
#include "GarbageCollectorKernelsLauncher.cuh"
#include "Macros.cuh"

class _RenderingKernelsLauncher
{
public:
    // renderScale: rendered pixels per screen pixel, greater than 1 if a picture is rendered with a higher resolution
    void drawImage(
        Settings const& settings,
        float2 rectUpperLeft,
        float2 rectLowerRight,
        int2 imageSize,
        float zoom,
        float renderScale,
        SimulationData data,
        RenderingData renderingData);
};
