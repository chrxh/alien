#pragma once

#include "EngineInterface/CudaSettings.h"
#include "EngineInterface/ShallowUpdateSelectionData.h"
#include "EngineInterface/SettingsForSimulation.h"
#include "EngineInterface/RenderData.h"

#include "Base.cuh"
#include "DataAccessKernels.cuh"
#include "Definitions.cuh"
#include "GarbageCollectorKernelsService.cuh"
#include "Macros.cuh"

class _RenderingKernelsService
{
public:
    _RenderingKernelsService();
    ~_RenderingKernelsService();

    void drawImage(
        SettingsForSimulation const& settings,
        float2 rectUpperLeft,
        float2 rectLowerRight,
        int2 imageSize,
        float zoom,
        SimulationData data,
        BufferData renderingData);

    NumRenderObjects getNumRenderObjects(SettingsForSimulation const& settings, SimulationData data);
    void extractObjectData(SettingsForSimulation const& settings, SimulationData data, BufferData& renderingData);

private:
    uint64_t* _numLineIndices;
};
