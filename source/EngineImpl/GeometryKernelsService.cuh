#pragma once

#include <Base/Singleton.h>

#include <EngineInterface/GeometryBuffers.h>

#include <EngineKernels/Base.cuh>
#include <EngineKernels/DataAccessKernels.cuh>
#include <EngineKernels/Definitions.cuh>
#include <EngineKernels/Macros.cuh>

class GeometryKernelsService
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(GeometryKernelsService);

public:
    void init();
    void shutdown();

    // Whether the geometry kernels can write directly into the shareable memory of the geometry buffers
    bool isSharedMemoryWorking(GeometryBuffers const& geometryBuffers);

    void correctPositionsForRendering(SettingsForSimulation const& settings, SimulationData data, RealRect const& visibleWorldRect);
    void restorePositions(SettingsForSimulation const& settings, SimulationData data);
    NumRenderObjects getNumRenderObjects(SettingsForSimulation const& settings, SimulationData data, RealRect const& visibleWorldRect);
    void
    extractObjectData(SettingsForSimulation const& settings, SimulationData data, CudaGeometryBuffers const& renderingData, RealRect const& visibleWorldRect);

private:
    GeometryKernelsService() = default;

    NumRenderObjects* _counters = nullptr;
};
