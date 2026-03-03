#pragma once

#include <Base/Singleton.h>

#include <EngineInterface/GeometryBuffers.h>

#include <EngineGpuKernels/Base.cuh>
#include <EngineGpuKernels/DataAccessKernels.cuh>
#include <EngineGpuKernels/Definitions.cuh>
#include <EngineGpuKernels/Macros.cuh>

class GeometryKernelsService
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(GeometryKernelsService);

public:
    void init();
    void shutdown();

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

    uint64_t* _numObjects = nullptr;
    uint64_t* _numLineIndices = nullptr;
    uint64_t* _numTriangleIndices = nullptr;
    uint64_t* _numSelectedConnectionVertices = nullptr;
    uint64_t* _numSelectedObjects = nullptr;
    uint64_t* _numAttackEventVertices = nullptr;
    uint64_t* _numDetonationEventVertices = nullptr;
    uint64_t* _numLocations = nullptr;
    uint64_t* _numBlurryParticles = nullptr;
};
