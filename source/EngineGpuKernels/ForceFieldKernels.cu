#include <EngineInterface/SimulationParameters.h>

#include "ConstantMemory.cuh"
#include "ForceFieldKernels.cuh"
#include "ParameterCalculator.cuh"

namespace
{
    __device__ float getHeight(BaseMap const& map, float2 const& pos, int const& index)
    {
        auto dist = map.getDistance(
            pos, float2{cudaSimulationParameters.layerPosition.layerValues[index].x, cudaSimulationParameters.layerPosition.layerValues[index].y});
        if (Orientation_Clockwise == cudaSimulationParameters.layerRadialForceFieldOrientation.layerValues[index]) {
            return sqrtf(dist) * cudaSimulationParameters.layerRadialForceFieldStrength.layerValues[index];
        } else {
            return -sqrtf(dist) * cudaSimulationParameters.layerRadialForceFieldStrength.layerValues[index];
        }
    }

    __device__ __inline__ float2 calcAcceleration(BaseMap const& map, float2 const& pos, int const& index)
    {
        switch (cudaSimulationParameters.layerForceFieldType.layerValues[index].value) {
        case ForceField_Radial: {
            auto baseValue = getHeight(map, pos, index);
            auto downValue = getHeight(map, pos + float2{0, 1}, index);
            auto rightValue = getHeight(map, pos + float2{1, 0}, index);
            float2 result{rightValue - baseValue, downValue - baseValue};
            result = Math::rotateClockwise(result, 90.0f + cudaSimulationParameters.layerRadialForceFieldDriftAngle.layerValues[index]);
            return result;
        }
        case ForceField_Central: {
            auto centerDirection = map.getCorrectedDirection(
                float2{cudaSimulationParameters.layerPosition.layerValues[index].x, cudaSimulationParameters.layerPosition.layerValues[index].y} - pos);
            return centerDirection * cudaSimulationParameters.layerCentralForceFieldStrength.layerValues[index]
                / (Math::lengthSquared(centerDirection) + 50.0f);
        }
        case ForceField_Linear: {
            auto centerDirection = Math::unitVectorOfAngle(cudaSimulationParameters.layerLinearForceFieldAngle.layerValues[index]);
            return centerDirection * cudaSimulationParameters.layerLinearForceFieldStrength.layerValues[index];
        }
        default:
            return {0, 0};
        }
    }

}

__global__ void cudaApplyForceFieldSettings(SimulationData data)
{
    float2 accelerations[MAX_LAYERS];

    auto calcResultingAcceleration = [&](float2 const& pos) {
        for (int i = 0; i < cudaSimulationParameters.numLayers; ++i) {
            if (cudaSimulationParameters.layerForceFieldType.layerValues[i].enabled) {
                accelerations[i] = calcAcceleration(data.objectMap, pos, i);
            } else {
                accelerations[i] = float2{0, 0};
            }
        }
        return ParameterCalculator::calcParameter(float2{0, 0}, accelerations, data, pos);
    };
    {
        auto& objects = data.entities.objects;
        auto partition = calcSystemThreadPartition(objects.getNumEntries());

        for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
            auto& object = objects.at(index);
            if (object->fixed) {
                continue;
            }
            object->vel += calcResultingAcceleration(object->pos);
        }
    }
    {
        auto& particles = data.entities.energies;
        auto partition = calcSystemThreadPartition(particles.getNumEntries());
        for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
            auto& particle = particles.at(index);
            particle->vel += calcResultingAcceleration(particle->pos);
        }
    }
}
