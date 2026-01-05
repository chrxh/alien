#pragma once

#include <cstdint>
#include <vector>

#include "GeometryBuffers.h"

// CPU-side geometry buffers for testing and non-rendering use cases
struct CpuGeometryBuffers
{
    std::vector<CellVertexData> cells;
    std::vector<EnergyParticleVertexData> energyParticles;
    std::vector<LocationVertexData> locations;
    std::vector<SelectedObjectVertexData> selectedObjects;
    std::vector<unsigned int> lineIndices;
    std::vector<unsigned int> triangleIndices;
    std::vector<ConnectionArrowVertexData> connectionArrows;
    std::vector<AttackEventVertexData> attackEvents;
    std::vector<DetonationEventVertexData> detonationEvents;

    NumRenderObjects getNumObjects() const
    {
        NumRenderObjects result;
        result.cells = cells.size();
        result.energyParticles = energyParticles.size();
        result.locations = locations.size();
        result.selectedObjects = selectedObjects.size();
        result.lineIndices = lineIndices.size();
        result.triangleIndices = triangleIndices.size();
        result.connectionArrowVertices = connectionArrows.size();
        result.attackEventVertices = attackEvents.size();
        result.detonationEventVertices = detonationEvents.size();
        return result;
    }
};
