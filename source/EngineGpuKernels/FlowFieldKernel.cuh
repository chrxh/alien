#pragma once

#include "Math.cuh"
#include "Map.cuh"
#include "SimulationData.cuh"

__device__ float getHeight(float2 const& pos, MapInfo const& mapInfo)
{
    float result = 0;
    for (int i = 0; i < cudaFlowFieldSettings.numCenters; ++i) {
        auto& radialFlow = cudaFlowFieldSettings.radialFlowCenters[i];
        auto dist = mapInfo.mapDistance(pos, float2{radialFlow.posX, radialFlow.posY});
        if (dist > radialFlow.radius) {
            dist = radialFlow.radius;
        }
        if (Orientation::Clockwise == radialFlow.orientation) {
            result += sqrtf(dist) * radialFlow.strength;
        } else {
            result -= sqrtf(dist) * radialFlow.strength;
        }
    }
    return result;
}

__device__ float2 calcVelocity(float2 const& pos, MapInfo const& mapInfo)
{
    auto baseValue = getHeight(pos, mapInfo);
    auto downValue = getHeight(pos + float2{0, 1}, mapInfo);
    auto rightValue = getHeight(pos + float2{1, 0}, mapInfo);
    float2 result{rightValue - baseValue, downValue - baseValue};
    Math::rotateQuarterClockwise(result);
    return result;
}


__global__ void applyFlowFieldSettings(SimulationData data)
{
    auto& cells = data.entities.cellPointers;
    auto partition = calcPartition(cells.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cells.at(index);
        cell->vel = cell->vel + calcVelocity(cell->absPos, data.cellMap);
    }
}

__global__ void applyFlowFieldSettingsKernel(SimulationData data)
{
    if (cudaFlowFieldSettings.active) {
        KERNEL_CALL(applyFlowFieldSettings, data);
    }
}