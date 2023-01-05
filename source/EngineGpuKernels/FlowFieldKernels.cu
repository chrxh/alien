#include "FlowFieldKernels.cuh"

#include "EngineInterface/SimulationParameters.h"

#include "ConstantMemory.cuh"
#include "SpotCalculator.cuh"

namespace
{
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
    auto partition = calcAllThreadsPartition(cells.getNumEntries());

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
