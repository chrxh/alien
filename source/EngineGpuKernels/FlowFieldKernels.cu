#include "FlowFieldKernels.cuh"

#include "EngineInterface/SimulationParameters.h"

#include "ConstantMemory.cuh"

namespace
{
    __device__ float getHeight(float2 const& pos, BaseMap const& mapInfo)
    {
        float result = 0;
        for (int i = 0; i < cudaSimulationParameters.numFlowCenters; ++i) {
            auto& radialFlow = cudaSimulationParameters.flowCenters[i];
            auto dist = mapInfo.getDistance(pos, float2{radialFlow.posX, radialFlow.posY});
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

    __device__ float2 calcVelocity(float2 const& pos, BaseMap const& mapInfo)
    {
        auto baseValue = getHeight(pos, mapInfo);
        auto downValue = getHeight(pos + float2{0, 1}, mapInfo);
        auto rightValue = getHeight(pos + float2{1, 0}, mapInfo);
        float2 result{rightValue - baseValue, downValue - baseValue};
        Math::rotateQuarterClockwise(result);
        return result;
    }


}

__global__ void cudaApplyFlowFieldSettings(SimulationData data)
{
    auto& cells = data.objects.cellPointers;
    auto partition = calcPartition(cells.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cells.at(index);
        if (cell->barrier) {
            continue;
        }
        cell->vel = cell->vel + calcVelocity(cell->absPos, data.cellMap);
        //auto d2 = Math::unitVectorOfAngle(data.numberGen1.random() * 360);
        //cell->vel = cell->vel + d2/3000;
        //auto d = float2{200, 100} - cell->absPos;
        //auto l = Math::length(d);
        //cell->vel = cell->vel + d/(l*l)/1000;
    }
}
