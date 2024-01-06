#include "FlowFieldKernels.cuh"

#include "EngineInterface/SimulationParameters.h"

#include "ConstantMemory.cuh"
#include "SpotCalculator.cuh"

namespace
{
    __device__ float getHeight(BaseMap const& map, float2 const& pos, SimulationParametersSpot const& spot)
    {
        auto dist = map.getDistance(pos, float2{spot.posX, spot.posY});
        if (Orientation_Clockwise == spot.flowData.radialFlow.orientation) {
            return sqrtf(dist) * spot.flowData.radialFlow.strength;
        } else {
            return -sqrtf(dist) * spot.flowData.radialFlow.strength;
        }
    }

    __device__ __inline__ float2 calcAcceleration(BaseMap const& map, float2 const& pos, int const& spotIndex)
    {
        auto const& spot = cudaSimulationParameters.spots[spotIndex];
        switch (spot.flowType) {
        case FlowType_Radial: {
            auto baseValue = getHeight(map, pos, spot);
            auto downValue = getHeight(map, pos + float2{0, 1}, spot);
            auto rightValue = getHeight(map, pos + float2{1, 0}, spot);
            float2 result{rightValue - baseValue, downValue - baseValue};
            result = Math::rotateClockwise(result, 90.0f + spot.flowData.radialFlow.driftAngle);
            return result;
        }
        case FlowType_Central: {
            auto centerDirection = map.getCorrectedDirection(float2{spot.posX, spot.posY} - pos);
            return centerDirection * spot.flowData.centralFlow.strength / (Math::lengthSquared(centerDirection) + 50.0f);
        }
        case FlowType_Linear: {
            auto centerDirection = Math::unitVectorOfAngle(spot.flowData.linearFlow.angle);
            return centerDirection * spot.flowData.linearFlow.strength;
        }
        default:
            return {0, 0};
        }

    }

}

__global__ void cudaApplyFlowFieldSettings(SimulationData data)
{
    float2 accelerations[MAX_SPOTS];
    {
        auto& cells = data.objects.cellPointers;
        auto partition = calcAllThreadsPartition(cells.getNumEntries());

        for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
            auto& cell = cells.at(index);
            if (cell->barrier) {
                continue;
            }
            int numFlowFields = 0;
            for (int i = 0; i < cudaSimulationParameters.numSpots; ++i) {

                if (cudaSimulationParameters.spots[i].flowType != FlowType_None) {
                    accelerations[numFlowFields] = calcAcceleration(data.cellMap, cell->pos, i);
                    ++numFlowFields;
                }
            }
            auto resultingAcceleration = SpotCalculator::calcResultingFlowField(data.cellMap, cell->pos, float2{0, 0}, accelerations);
            cell->shared1 += resultingAcceleration;
        }
    }
    {
        auto& particles = data.objects.particlePointers;
        auto partition = calcAllThreadsPartition(particles.getNumEntries());
        for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
            auto& particle = particles.at(index);
            int numFlowFields = 0;
            for (int i = 0; i < cudaSimulationParameters.numSpots; ++i) {

                if (cudaSimulationParameters.spots[i].flowType != FlowType_None) {
                    accelerations[numFlowFields] = calcAcceleration(data.cellMap, particle->absPos, i);
                    ++numFlowFields;
                }
            }
            auto resultingAcceleration = SpotCalculator::calcResultingFlowField(data.cellMap, particle->absPos, float2{0, 0}, accelerations);
            particle->vel += resultingAcceleration;
        }
    }
}
