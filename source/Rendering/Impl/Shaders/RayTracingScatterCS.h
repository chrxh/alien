#pragma once

#include <string_view>

namespace Shaders
{
    // Writes the lights into the ranges of their grid cells. The criteria have to match the counting in RayTracingPrepareCS.
    std::string_view const RayTracingScatterCS = R"(
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

void main()
{
    uint index = gl_GlobalInvocationID.x;
    if (index >= frame.counts.x) {
        return;
    }
    Sphere sphere = spheres[index];
    if (sphere.hueMaterial.w == Material_Inactive || !isLight(sphere)) {
        return;
    }
    uint cellIndex = getLightCellIndex(getLightCell(sphere.centerRadius.xy));
    if (cellIndex != NoEntry) {
        cellEntries[atomicAdd(cellCursors[cellIndex], 1u)] = index;
    }
}
)";
}
