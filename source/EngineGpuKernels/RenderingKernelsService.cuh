#pragma once

#include "EngineInterface/GeometryBuffers.h"

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
        CudaGeometryBuffers renderingData);

    NumRenderObjects getNumRenderObjects(SettingsForSimulation const& settings, SimulationData data);
    void extractObjectData(SettingsForSimulation const& settings, SimulationData data, CudaGeometryBuffers& renderingData);

private:
    uint64_t* _numLineIndices;
};
