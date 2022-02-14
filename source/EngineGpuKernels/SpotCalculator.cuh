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
            float2 spotPos = {cudaSimulationParametersSpots.spots[0].posX, cudaSimulationParametersSpots.spots[0].posY};
            auto delta = spotPos - worldPos;
            map.correctDirection(delta);
            auto weight = calcWeight(delta, 0);
            return mix(baseValue, spotValue1, weight);
        } else if (2 == cudaSimulationParametersSpots.numSpots) {
            float2 spotPos1 = {cudaSimulationParametersSpots.spots[0].posX, cudaSimulationParametersSpots.spots[0].posY};
            float2 spotPos2 = {cudaSimulationParametersSpots.spots[1].posX, cudaSimulationParametersSpots.spots[1].posY};

            auto delta1 = spotPos1 - worldPos;
            map.correctDirection(delta1);
            auto delta2 = spotPos2 - worldPos;
            map.correctDirection(delta2);

            auto weight1 = calcWeight(delta1, 0);
            auto weight2 = calcWeight(delta2, 1);

            return mix(baseValue, spotValue1, spotValue2, weight1, weight2);
        }
        return baseValue;
    }

    __device__ __inline__ static float calcWeight(float2 const& delta, int const& spotIndex)
    {
        if (cudaSimulationParametersSpots.spots[spotIndex].shape == SpotShape::Rectangular) {
            return calcWeightForRectSpot(delta, spotIndex);
        }
        return calcWeightForCircularSpot(delta, spotIndex);
    }

    __device__ __inline__ static float calcWeightForCircularSpot(float2 const& delta, int const& spotIndex)
    {
        auto distance = Math::length(delta);
        auto coreRadius = cudaSimulationParametersSpots.spots[spotIndex].coreRadius;
        auto fadeoutRadius = cudaSimulationParametersSpots.spots[spotIndex].fadeoutRadius + 1;
        return distance < coreRadius ? 0.0f : min(1.0f, (distance - coreRadius) / fadeoutRadius);
    }

    __device__ __inline__ static float calcWeightForRectSpot(float2 const& delta, int const& spotIndex)
    {
        auto const& spot = cudaSimulationParametersSpots.spots[spotIndex];
        float result = 0;
        if (abs(delta.x) > spot.width / 2 || abs(delta.y) > spot.height / 2) {
            float2 distanceFromRect = {max(0.0f, abs(delta.x) - spot.width / 2), max(0.0f, abs(delta.y) - spot.height / 2)};
            result = min(1.0f, Math::length(distanceFromRect) / (spot.fadeoutRadius + 1));
        }
        return result;
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