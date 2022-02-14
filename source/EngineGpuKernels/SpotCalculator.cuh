#pragma once

#include "cuda_runtime_api.h"

#include "EngineInterface/SimulationParametersSpotValues.h"
#include "ConstantMemory.cuh"

class SpotCalculator
{
public:
    __device__ __inline__ static float calcParameter(float SimulationParametersSpotValues::*value, SimulationData const& data, float2 const& worldPos)
    {
        return calcResultingValue(
            data.cellMap,
            worldPos,
            cudaSimulationParameters.spotValues.*value,
            cudaSimulationParametersSpots.spots[0].values.*value,
            cudaSimulationParametersSpots.spots[1].values.*value);
    }

    __device__ __inline__ static float3
    calcColor(BaseMap const& map, float2 const& worldPos, float3 const& baseColor, float3 const& spotColor1, float3 const& spotColor2)
    {
        return calcResultingValue(map, worldPos, baseColor, spotColor1, spotColor2);
    }

private:
    template<typename T>
    __device__ __inline__ static T calcResultingValue(BaseMap const& map, float2 const& worldPos, T const& baseValue, T const& spotValue1, T const& spotValue2)
    {
        if (1 == cudaSimulationParametersSpots.numSpots) {
            auto distance = map.getDistance(worldPos, {cudaSimulationParametersSpots.spots[0].posX, cudaSimulationParametersSpots.spots[0].posY});
            auto coreRadius = cudaSimulationParametersSpots.spots[0].coreRadius;
            auto fadeoutRadius = cudaSimulationParametersSpots.spots[0].fadeoutRadius + 1;
            auto factor = distance < coreRadius ? 0.0f : min(1.0f, (distance - coreRadius) / fadeoutRadius);
            return mix(baseValue, spotValue1, factor);
        }
        if (2 == cudaSimulationParametersSpots.numSpots) {
            auto distance1 = map.getDistance(worldPos, {cudaSimulationParametersSpots.spots[0].posX, cudaSimulationParametersSpots.spots[0].posY});
            auto distance2 = map.getDistance(worldPos, {cudaSimulationParametersSpots.spots[1].posX, cudaSimulationParametersSpots.spots[1].posY});

            auto coreRadius1 = cudaSimulationParametersSpots.spots[0].coreRadius;
            auto fadeoutRadius1 = cudaSimulationParametersSpots.spots[0].fadeoutRadius + 1;
            auto factor1 = distance1 < coreRadius1 ? 0.0f : min(1.0f, (distance1 - coreRadius1) / fadeoutRadius1);
            auto coreRadius2 = cudaSimulationParametersSpots.spots[1].coreRadius;
            auto fadeoutRadius2 = cudaSimulationParametersSpots.spots[1].fadeoutRadius + 1;
            auto factor2 = distance2 < coreRadius2 ? 0.0f : min(1.0f, (distance2 - coreRadius2) / fadeoutRadius2);

            return mix(baseValue, spotValue1, spotValue2, factor1, factor2);
        }
        return baseValue;
    }

    __device__ __inline__ static float mix(float const& a, float const& b, float factor) { return a * factor + b * (1 - factor); }

    __device__ __inline__ static float3 mix(float3 const& a, float3 const& b, float factor)
    {
        return float3{a.x * factor + b.x * (1 - factor), a.y * factor + b.y * (1 - factor), a.z * factor + b.z * (1 - factor)};
    }

    __device__ __inline__ static float mix(float const& a, float const& b, float const& c, float factor1, float factor2)
    {
        float weight1 = factor1 * factor2;
        float weight2 = 1 - factor1;
        float weight3 = 1 - factor2;
        float sum = weight1 + weight2 + weight3;
        weight1 /= sum;
        weight2 /= sum;
        weight3 /= sum;
        return a * weight1 + b * weight2 + c * weight3;
    }

    __device__ __inline__ static float3 mix(float3 const& a, float3 const& b, float3 const& c, float factor1, float factor2)
    {
        float weight1 = factor1 * factor2;
        float weight2 = 1 - factor1;
        float weight3 = 1 - factor2;
        float sum = weight1 + weight2 + weight3;
        weight1 /= sum;
        weight2 /= sum;
        weight3 /= sum;
        return float3{
            a.x * weight1 + b.x * weight2 + c.x * weight3, a.y * weight1 + b.y * weight2 + c.y * weight3, a.z * weight1 + b.z * weight2 + c.z * weight3};
    }
};