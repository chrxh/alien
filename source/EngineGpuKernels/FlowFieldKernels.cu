#include "FlowFieldKernels.cuh"

#include "EngineInterface/SimulationParameters.h"

#include "ConstantMemory.cuh"
#include "ZoneCalculator.cuh"

namespace
{
    __device__ float getHeight(BaseMap const& map, float2 const& pos, int const& zoneIndex)
    {
        auto const& zone = cudaSimulationParameters.zone[zoneIndex];

        auto dist =
            map.getDistance(pos, float2{cudaSimulationParameters.zonePosition.zoneValues[zoneIndex].x, cudaSimulationParameters.zonePosition.zoneValues[zoneIndex].y});
        if (Orientation_Clockwise == zone.flow.alternatives.radialFlow.orientation) {
            return sqrtf(dist) * zone.flow.alternatives.radialFlow.strength;
        } else {
            return -sqrtf(dist) * zone.flow.alternatives.radialFlow.strength;
        }
    }

    __device__ __inline__ float2 calcAcceleration(BaseMap const& map, float2 const& pos, int const& zoneIndex)
    {
        auto const& zone = cudaSimulationParameters.zone[zoneIndex];
        switch (zone.flow.type) {
        case FlowType_Radial: {
            auto baseValue = getHeight(map, pos, zoneIndex);
            auto downValue = getHeight(map, pos + float2{0, 1}, zoneIndex);
            auto rightValue = getHeight(map, pos + float2{1, 0}, zoneIndex);
            float2 result{rightValue - baseValue, downValue - baseValue};
            result = Math::rotateClockwise(result, 90.0f + zone.flow.alternatives.radialFlow.driftAngle);
            return result;
        }
        case FlowType_Central: {
            auto centerDirection = map.getCorrectedDirection(
                float2{cudaSimulationParameters.zonePosition.zoneValues[zoneIndex].x, cudaSimulationParameters.zonePosition.zoneValues[zoneIndex].y} - pos);
            return centerDirection * zone.flow.alternatives.centralFlow.strength / (Math::lengthSquared(centerDirection) + 50.0f);
        }
        case FlowType_Linear: {
            auto centerDirection = Math::unitVectorOfAngle(zone.flow.alternatives.linearFlow.angle);
            return centerDirection * zone.flow.alternatives.linearFlow.strength;
        }
        default:
            return {0, 0};
        }

    }

}

__global__ void cudaApplyFlowFieldSettings(SimulationData data)
{
    float2 accelerations[MAX_ZONES];
    {
        auto& cells = data.objects.cellPointers;
        auto partition = calcAllThreadsPartition(cells.getNumEntries());

        for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
            auto& cell = cells.at(index);
            if (cell->barrier) {
                continue;
            }
            int numFlowFields = 0;
            for (int i = 0; i < cudaSimulationParameters.numZones.value; ++i) {

                if (cudaSimulationParameters.zone[i].flow.type != FlowType_None) {
                    accelerations[numFlowFields] = calcAcceleration(data.cellMap, cell->pos, i);
                    ++numFlowFields;
                }
            }
            auto resultingAcceleration = ZoneCalculator::calcResultingFlowField(data.cellMap, cell->pos, float2{0, 0}, accelerations);
            cell->shared1 += resultingAcceleration;
        }
    }
    {
        auto& particles = data.objects.particlePointers;
        auto partition = calcAllThreadsPartition(particles.getNumEntries());
        for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
            auto& particle = particles.at(index);
            int numFlowFields = 0;
            for (int i = 0; i < cudaSimulationParameters.numZones.value; ++i) {

                if (cudaSimulationParameters.zone[i].flow.type != FlowType_None) {
                    accelerations[numFlowFields] = calcAcceleration(data.cellMap, particle->absPos, i);
                    ++numFlowFields;
                }
            }
            auto resultingAcceleration = ZoneCalculator::calcResultingFlowField(data.cellMap, particle->absPos, float2{0, 0}, accelerations);
            particle->vel += resultingAcceleration;
        }
    }
}
