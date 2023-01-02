#include "FlowFieldKernels.cuh"

#include "EngineInterface/SimulationParameters.h"

#include "ConstantMemory.cuh"
#include "SpotCalculator.cuh"

namespace
{
    //__device__ float getHeight(float2 const& pos, BaseMap const& mapInfo)
    //{
    //    float result = 0;
    //    for (int i = 0; i < cudaSimulationParameters.numFlowCenters; ++i) {
    //        auto& radialFlow = cudaSimulationParameters.flowCenters[i];
    //        auto dist = mapInfo.getDistance(pos, float2{radialFlow.posX, radialFlow.posY});
    //        if (dist > radialFlow.radius) {
    //            dist = radialFlow.radius;
    //        }
    //        if (Orientation::Clockwise == radialFlow.orientation) {
    //            result += sqrtf(dist) * radialFlow.strength;
    //        } else {
    //            result -= sqrtf(dist) * radialFlow.strength;
    //        }
    //    }
    //    return result;
    //}

    //__device__ float2 calcVelocity(float2 const& pos, BaseMap const& mapInfo)
    //{
    //    auto baseValue = getHeight(pos, mapInfo);
    //    auto downValue = getHeight(pos + float2{0, 1}, mapInfo);
    //    auto rightValue = getHeight(pos + float2{1, 0}, mapInfo);
    //    float2 result{rightValue - baseValue, downValue - baseValue};
    //    Math::rotateQuarterClockwise(result);
    //    return result;
    //}

    __device__ float getHeight(BaseMap const& map, float2 const& pos, int const& spotIndex)
    {
        auto const& spot = cudaSimulationParameters.spots[spotIndex];
        auto dist = map.getDistance(pos, float2{spot.posX, spot.posY});
        if (Orientation_Clockwise == spot.flowData.radialFlow.orientation) {
            return sqrtf(dist) * spot.flowData.radialFlow.strength;
        } else {
            return -sqrtf(dist) * spot.flowData.radialFlow.strength;
        }
    }

    __device__ __inline__ float2 calcAcceleration(BaseMap const& map, float2 const& pos, int const& spotIndex)
    {
        if (cudaSimulationParameters.spots[spotIndex].flowType == FlowType_None) {
            return {0, 0};
        }
        auto baseValue = getHeight(map, pos, spotIndex);
        auto downValue = getHeight(map, pos + float2{0, 1}, spotIndex);
        auto rightValue = getHeight(map, pos + float2{1, 0}, spotIndex);
        float2 result{rightValue - baseValue, downValue - baseValue};
        Math::rotateQuarterClockwise(result);
        return result;
    }

}

__global__ void cudaApplyFlowFieldSettings(SimulationData data)
{
    auto& cells = data.objects.cellPointers;
    auto partition = calcPartition(cells.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

    float2 accelerations[MAX_SPOTS];
    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cells.at(index);
        if (cell->barrier) {
            continue;
        }
        for (int i = 0; i < cudaSimulationParameters.numSpots; ++i) {
            accelerations[i] = calcAcceleration(data.cellMap, cell->absPos, i);
        }
        auto resultingAcceleration =
            SpotCalculator::calcResultingValue(data.cellMap, cell->absPos, float2{0, 0}, accelerations);
        cell->vel = cell->vel + resultingAcceleration;
    }
}
