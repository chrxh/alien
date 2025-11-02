#pragma once

#include <EngineInterface/GeometryBuffers.h>

#include "Base.cuh"
#include "DataAccessKernels.cuh"
#include "Definitions.cuh"
#include "GarbageCollectorKernelsService.cuh"
#include "Macros.cuh"

class _GeometryKernelsService
{
public:
    _GeometryKernelsService();
    ~_GeometryKernelsService();

    void correctPositionsForRendering(SettingsForSimulation const& settings, SimulationData data, RealRect const& visibleWorldRect);
    void restorePositions(SettingsForSimulation const& settings, SimulationData data);
    NumRenderObjects getNumRenderObjects(SettingsForSimulation const& settings, SimulationData data, RealRect const& visibleWorldRect);
    void extractObjectData(SettingsForSimulation const& settings, SimulationData data, CudaGeometryBuffers& renderingData, RealRect const& visibleWorldRect);

private:
    uint64_t* _numLineIndices;
    uint64_t* _numTriangleIndices;
    uint64_t* _numSelectedConnectionVertices;
    uint64_t* _numSelectedObjects;
    uint64_t* _numAttackEventVertices;
    uint64_t* _numDetonationEventVertices;
    uint64_t* _numLocations;
};
