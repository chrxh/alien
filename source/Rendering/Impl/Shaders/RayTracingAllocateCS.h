#pragma once

#include <string_view>

namespace Shaders
{
    // Reserves a range of the entry buffer for every cell of the light grid. A work group reserves the ranges of its cells with a single
    // atomic operation.
    std::string_view const RayTracingAllocateCS = R"(
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

shared uint groupCount;
shared uint groupOffset;

void main()
{
    uint cellIndex = gl_GlobalInvocationID.x;
    if (gl_LocalInvocationIndex == 0u) {
        groupCount = 0u;
    }
    memoryBarrierShared();
    barrier();

    uint count = cellIndex < frame.counts.w ? cellCounts[cellIndex] : 0u;
    uint localOffset = atomicAdd(groupCount, count);
    memoryBarrierShared();
    barrier();

    if (gl_LocalInvocationIndex == 0u) {
        groupOffset = atomicAdd(allocatedEntries, groupCount);
    }
    memoryBarrierShared();
    barrier();

    if (cellIndex < frame.counts.w) {
        cellOffsets[cellIndex] = groupOffset + localOffset;
        cellCursors[cellIndex] = groupOffset + localOffset;
    }
}
)";
}
