#pragma once

#include <string_view>

namespace Shaders
{
    // Collects the energy of all objects in the energy grid, from which the regions with a high energy density radiate, see the energy light
    // in Shaders/RayTracingCommonCS.h
    std::string_view const RayTracingEnergyGridCS = R"(
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

void main()
{
    uint index = gl_GlobalInvocationID.x;
    if (index >= frame.counts.x) {
        return;
    }
    ObjectVertex object = objects[index];
    float energy = getObjectEnergy(object);
    if (energy <= 0.0) {
        return;
    }
    vec2 coord = getEnergyCellCoord(vec2(object.pos[0], object.pos[1]));
    ivec2 lowerCell = ivec2(floor(coord));
    vec2 fraction = coord - vec2(lowerCell);
    for (int i = 0; i < 4; ++i) {
        ivec2 offset = ivec2(i & 1, i >> 1);
        ivec2 cell = lowerCell + offset;
        if (isInEnergyGrid(cell)) {
            vec2 axisWeights = mix(1.0 - fraction, fraction, vec2(offset));
            float cellEnergy = axisWeights.x * axisWeights.y * energy * EnergyFixedPointScale;
            vec2 shiftedOffset = coord - vec2(cell) + 1.0;
            uint entry = getEnergyCellIndex(cell) * EnergyCellStride;
            atomicAdd(energyGrid[entry], uint(cellEnergy + 0.5));
            atomicAdd(energyGrid[entry + 1u], uint(cellEnergy * shiftedOffset.x + 0.5));
            atomicAdd(energyGrid[entry + 2u], uint(cellEnergy * shiftedOffset.y + 0.5));
        }
    }
}
)";
}
