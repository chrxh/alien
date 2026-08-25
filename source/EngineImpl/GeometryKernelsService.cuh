#pragma once

#include <optional>

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

    bool checkForInterop();

    void correctPositionsForRendering(SettingsForSimulation const& settings, SimulationData data, RealRect const& visibleWorldRect);
    void restorePositions(SettingsForSimulation const& settings, SimulationData data);
    NumRenderObjects getNumRenderObjects(SettingsForSimulation const& settings, SimulationData data, RealRect const& visibleWorldRect);
    void extractObjectData(
        SettingsForSimulation const& settings,
        SimulationData data,
        CudaGeometryBuffers& renderingData,
        RealRect const& visibleWorldRect,
        bool useInterop);

private:
    GeometryKernelsService() = default;

    NumRenderObjects* _counters = nullptr;
    std::optional<bool> _interopUsable;
};
