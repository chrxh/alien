#pragma once

#include <Base/Interface/Singleton.h>

#include <Engine/Interface/GeometryBuffers.h>

#include <Engine/Kernels/Base.cuh>
#include <Engine/Kernels/DataAccessKernels.cuh>
#include <Engine/Kernels/Definitions.cuh>
#include <Engine/Kernels/Macros.cuh>

class GeometryKernelsService
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(GeometryKernelsService);

public:
    void init();
    void shutdown();

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
