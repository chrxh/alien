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

    __device__ __inline__ float perlinFade(float t)
    {
        return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
    }

    __device__ __inline__ int perlinHash(int x)
    {
        x = ((x >> 16) ^ x) * 0x45d9f3b;
        x = ((x >> 16) ^ x) * 0x45d9f3b;
        x = (x >> 16) ^ x;
        return x;
    }

    __device__ __inline__ float2 perlinGradient(int ix, int iy)
    {
        int h = perlinHash(perlinHash(ix) + iy) & 3;
        switch (h) {
        case 0:
            return {1.0f, 0.0f};
        case 1:
            return {-1.0f, 0.0f};
        case 2:
            return {0.0f, 1.0f};
        default:
            return {0.0f, -1.0f};
        }
    }

    __device__ __inline__ float perlinNoise(float x, float y)
    {
        int ix = __float2int_rd(x);
        int iy = __float2int_rd(y);
        float fx = x - ix;
        float fy = y - iy;

        float u = perlinFade(fx);
        float v = perlinFade(fy);

        float2 g00 = perlinGradient(ix, iy);
        float2 g10 = perlinGradient(ix + 1, iy);
        float2 g01 = perlinGradient(ix, iy + 1);
        float2 g11 = perlinGradient(ix + 1, iy + 1);

        float n00 = g00.x * fx + g00.y * fy;
        float n10 = g10.x * (fx - 1.0f) + g10.y * fy;
        float n01 = g01.x * fx + g01.y * (fy - 1.0f);
        float n11 = g11.x * (fx - 1.0f) + g11.y * (fy - 1.0f);

        float nx0 = n00 + u * (n10 - n00);
        float nx1 = n01 + u * (n11 - n01);

        return nx0 + v * (nx1 - nx0);
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
        case ForceField_PerlinNoise: {
            auto layerPos = cudaSimulationParameters.layerPosition.layerValues[index];
            float2 relPos{pos.x - layerPos.x, pos.y - layerPos.y};
            map.correctDirection(relPos);
            auto scale = 0.05f;
            auto sx = relPos.x * scale;
            auto sy = relPos.y * scale;
            auto baseValue = perlinNoise(sx, sy);
            auto rightValue = perlinNoise(sx + scale, sy);
            auto downValue = perlinNoise(sx, sy + scale);
            auto strength = cudaSimulationParameters.layerPerlinNoiseForceFieldStrength.layerValues[index];
            return float2{rightValue - baseValue, downValue - baseValue} * strength;
        }
        default:
            return {0, 0};
        }
    }

}

__global__ void cudaApplyForceFields(SimulationData data)
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
