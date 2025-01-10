﻿#pragma once

#include "EngineInterface/GpuSettings.h"
#include "EngineInterface/ShallowUpdateSelectionData.h"
#include "EngineInterface/SettingsForSimulation.h"

#include "Base.cuh"
#include "DataAccessKernels.cuh"
#include "Definitions.cuh"
#include "GarbageCollectorKernelsLauncher.cuh"
#include "Macros.cuh"

class _RenderingKernelsLauncher
{
public:
    void drawImage(
        SettingsForSimulation const& settings,
        float2 rectUpperLeft,
        float2 rectLowerRight,
        int2 imageSize,
        float zoom,
        SimulationData data,
        RenderingData renderingData);
};
