#include "ForceFieldKernels.cuh"

#include "EngineInterface/SimulationParameters.h"

#include "ConstantMemory.cuh"
#include "ZoneCalculator.cuh"

namespace
{
    __device__ float getHeight(BaseMap const& map, float2 const& pos, int const& zoneIndex)
    {
        auto dist =
            map.getDistance(pos, float2{cudaSimulationParameters.zonePosition.zoneValues[zoneIndex].x, cudaSimulationParameters.zonePosition.zoneValues[zoneIndex].y});
        if (Orientation_Clockwise == cudaSimulationParameters.zoneRadialForceFieldOrientation.zoneValues[zoneIndex]) {
            return sqrtf(dist) * cudaSimulationParameters.zoneRadialForceFieldStrength.zoneValues[zoneIndex];
        } else {
            return -sqrtf(dist) * cudaSimulationParameters.zoneRadialForceFieldStrength.zoneValues[zoneIndex];
        }
    }

    __device__ __inline__ float2 calcAcceleration(BaseMap const& map, float2 const& pos, int const& zoneIndex)
    {
        switch (cudaSimulationParameters.zoneForceFieldType.zoneValues[zoneIndex]) {
        case ForceField_Radial: {
            auto baseValue = getHeight(map, pos, zoneIndex);
            auto downValue = getHeight(map, pos + float2{0, 1}, zoneIndex);
            auto rightValue = getHeight(map, pos + float2{1, 0}, zoneIndex);
            float2 result{rightValue - baseValue, downValue - baseValue};
            result = Math::rotateClockwise(
                result, 90.0f + cudaSimulationParameters.zoneRadialForceFieldDriftAngle.zoneValues[zoneIndex]);
            return result;
        }
        case ForceField_Central: {
            auto centerDirection = map.getCorrectedDirection(
                float2{cudaSimulationParameters.zonePosition.zoneValues[zoneIndex].x, cudaSimulationParameters.zonePosition.zoneValues[zoneIndex].y} - pos);
            return centerDirection * cudaSimulationParameters.zoneCentralForceFieldStrength.zoneValues[zoneIndex]
                / (Math::lengthSquared(centerDirection) + 50.0f);
        }
        case ForceField_Linear: {
            auto centerDirection = Math::unitVectorOfAngle(cudaSimulationParameters.zoneLinearForceFieldAngle.zoneValues[zoneIndex]);
            return centerDirection * cudaSimulationParameters.zoneLinearForceFieldStrength.zoneValues[zoneIndex];
        }
        default:
            return {0, 0};
        }

    }

}

__global__ void cudaApplyForceFieldSettings(SimulationData data)
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

                if (cudaSimulationParameters.zoneForceFieldType.zoneValues[i] != ForceField_None) {
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

                if (cudaSimulationParameters.zoneForceFieldType.zoneValues[i] != ForceField_None) {
                    accelerations[numFlowFields] = calcAcceleration(data.cellMap, particle->absPos, i);
                    ++numFlowFields;
                }
            }
            auto resultingAcceleration = ZoneCalculator::calcResultingFlowField(data.cellMap, particle->absPos, float2{0, 0}, accelerations);
            particle->vel += resultingAcceleration;
        }
    }
}
