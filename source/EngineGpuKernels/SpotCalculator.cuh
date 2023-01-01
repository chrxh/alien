#pragma once

#include "cuda_runtime_api.h"

#include "EngineInterface/SimulationParametersSpotValues.h"
#include "ConstantMemory.cuh"
#include "Swap.cuh"

class SpotCalculator
{
public:
    __device__ __inline__ static float calcParameter(float SimulationParametersSpotValues::*value, SimulationData const& data, float2 const& worldPos)
    {
        int spotIndex1, spotIndex2;
        getNearbySpots(data.cellMap, worldPos, spotIndex1, spotIndex2);

        return calcResultingValue(
            data.cellMap,
            worldPos,
            cudaSimulationParameters.baseValues.*value,
            cudaSimulationParameters.spots[spotIndex1].values.*value,
            cudaSimulationParameters.spots[spotIndex2].values.*value,
            spotIndex1,
            spotIndex2);
    }

    __device__ __inline__ static int calcParameter(int SimulationParametersSpotValues::*value, SimulationData const& data, float2 const& worldPos)
    {
        int spotIndex1, spotIndex2;
        getNearbySpots(data.cellMap, worldPos, spotIndex1, spotIndex2);

        return toInt(calcResultingValue(
            data.cellMap,
            worldPos,
            toFloat(cudaSimulationParameters.baseValues.*value),
            toFloat(cudaSimulationParameters.spots[spotIndex1].values.*value),
            toFloat(cudaSimulationParameters.spots[spotIndex2].values.*value),
            spotIndex1,
            spotIndex2));
    }

    __device__ __inline__ static float calcColorMatrix(int color, int otherColor, SimulationData const& data, float2 const& worldPos)
    {
        int spotIndex1, spotIndex2;
        getNearbySpots(data.cellMap, worldPos, spotIndex1, spotIndex2);

        return calcResultingValue(
            data.cellMap,
            worldPos,
            cudaSimulationParameters.baseValues.cellFunctionAttackerFoodChainColorMatrix[color][otherColor],
            cudaSimulationParameters.spots[spotIndex1].values.cellFunctionAttackerFoodChainColorMatrix[color][otherColor],
            cudaSimulationParameters.spots[spotIndex2].values.cellFunctionAttackerFoodChainColorMatrix[color][otherColor],
            spotIndex1,
            spotIndex2);
    }

    __device__ __inline__ static int calcColorTransitionDuration(int color, SimulationData const& data, float2 const& worldPos)
    {
        int spotIndex1, spotIndex2;
        getNearbySpots(data.cellMap, worldPos, spotIndex1, spotIndex2);

        return toInt(calcResultingValue(
            data.cellMap,
            worldPos,
            toFloat(cudaSimulationParameters.baseValues.cellColorTransitionDuration[color]),
            toFloat(cudaSimulationParameters.spots[spotIndex1].values.cellColorTransitionDuration[color]),
            toFloat(cudaSimulationParameters.spots[spotIndex2].values.cellColorTransitionDuration[color]),
            spotIndex1,
            spotIndex2));
    }

    __device__ __inline__ static int calcColorTransitionTargetColor(int color, SimulationData const& data, float2 const& worldPos)
    {
        int spotIndex1, spotIndex2;
        getNearbySpots(data.cellMap, worldPos, spotIndex1, spotIndex2);

        return toInt(
            calcResultingValue(
            data.cellMap,
            worldPos,
            toFloat(cudaSimulationParameters.baseValues.cellColorTransitionTargetColor[color]),
                toFloat(cudaSimulationParameters.spots[spotIndex1].values.cellColorTransitionTargetColor[color]),
                toFloat(cudaSimulationParameters.spots[spotIndex2].values.cellColorTransitionTargetColor[color]),
                spotIndex1,
                spotIndex2)
            + 0.5f);
    }

    __device__ __inline__ static void getNearbySpots(BaseMap const& map, float2 const& worldPos, int& spotIndex1, int& spotIndex2) 
    {
        if (cudaSimulationParameters.numSpots <= 2) {
            spotIndex1 = 0;
            spotIndex2 = 1;
            return;
        }
        float smallestSpotDistance = map.getDistance(worldPos, {cudaSimulationParameters.spots[0].posX, cudaSimulationParameters.spots[0].posY});
        float secondSmallestSpotDistance = map.getDistance(worldPos, {cudaSimulationParameters.spots[1].posX, cudaSimulationParameters.spots[1].posY});
        if (secondSmallestSpotDistance < smallestSpotDistance) {
            swap(secondSmallestSpotDistance, smallestSpotDistance);
            spotIndex1 = 1;
            spotIndex2 = 0;
        } else {
            spotIndex1 = 0;
            spotIndex2 = 1;
        }

         for (int i = 2; i < cudaSimulationParameters.numSpots; ++i) {
            float spotDistance = map.getDistance(worldPos, {cudaSimulationParameters.spots[i].posX, cudaSimulationParameters.spots[i].posY});
            if (spotDistance <= smallestSpotDistance) {
                spotIndex2 = spotIndex1;
                secondSmallestSpotDistance = smallestSpotDistance;
                spotIndex1 = i;
                smallestSpotDistance = spotDistance;
            } else if (spotDistance <= secondSmallestSpotDistance) {
                spotIndex2 = i;
                secondSmallestSpotDistance = spotDistance;
            }
        }
    }

    template <typename T>
    __device__ __inline__ static T calcResultingValue(
        BaseMap const& map,
        float2 const& worldPos,
        T const& baseValue,
        T const& spotValue1,
        T const& spotValue2,
        int const& spotIndex1,
        int const& spotIndex2)
    {
        if (0 == cudaSimulationParameters.numSpots) {
            return baseValue;
        } else if (1 == cudaSimulationParameters.numSpots) {
            float2 spotPos = {cudaSimulationParameters.spots[0].posX, cudaSimulationParameters.spots[0].posY};
            auto delta = spotPos - worldPos;
            map.correctDirection(delta);
            auto weight = calcWeight(delta, 0);
            return mix(baseValue, spotValue1, weight);
        } else {
            float2 spotPos1 = {cudaSimulationParameters.spots[spotIndex1].posX, cudaSimulationParameters.spots[spotIndex1].posY};
            float2 spotPos2 = {cudaSimulationParameters.spots[spotIndex2].posX, cudaSimulationParameters.spots[spotIndex2].posY};

            auto delta1 = spotPos1 - worldPos;
            map.correctDirection(delta1);
            auto delta2 = spotPos2 - worldPos;
            map.correctDirection(delta2);

            auto weight1 = calcWeight(delta1, spotIndex1);
            auto weight2 = calcWeight(delta2, spotIndex2);

            return mix(baseValue, spotValue1, spotValue2, weight1, weight2);
        }
        return baseValue;
    }

    __device__ __inline__ static float calcWeight(float2 const& delta, int const& spotIndex)
    {
        if (cudaSimulationParameters.spots[spotIndex].shapeType == ShapeType_Rectangular) {
            return calcWeightForRectSpot(delta, spotIndex);
        }
        return calcWeightForCircularSpot(delta, spotIndex);
    }

private:

    __device__ __inline__ static float calcWeightForCircularSpot(float2 const& delta, int const& spotIndex)
    {
        auto distance = Math::length(delta);
        auto coreRadius = cudaSimulationParameters.spots[spotIndex].shapeData.circularSpot.coreRadius;
        auto fadeoutRadius = cudaSimulationParameters.spots[spotIndex].fadeoutRadius + 1;
        return distance < coreRadius ? 0.0f : min(1.0f, (distance - coreRadius) / fadeoutRadius);
    }

    __device__ __inline__ static float calcWeightForRectSpot(float2 const& delta, int const& spotIndex)
    {
        auto const& spot = cudaSimulationParameters.spots[spotIndex];
        float result = 0;
        if (abs(delta.x) > spot.shapeData.rectangularSpot.width / 2 || abs(delta.y) > spot.shapeData.rectangularSpot.height / 2) {
            float2 distanceFromRect = {
                max(0.0f, abs(delta.x) - spot.shapeData.rectangularSpot.width / 2), max(0.0f, abs(delta.y) - spot.shapeData.rectangularSpot.height / 2)};
            result = min(1.0f, Math::length(distanceFromRect) / (spot.fadeoutRadius + 1));
        }
        return result;
    }

    __device__ __inline__ static float mix(float const& a, float const& b, float factor) { return a * factor + b * (1 - factor); }

    __device__ __inline__ static float2 mix(float2 const& a, float2 const& b, float factor)
    {
        return {a.x * factor + b.x * (1 - factor), a.y * factor + b.y * (1 - factor)};
    }

    __device__ __inline__ static float3 mix(float3 const& a, float3 const& b, float factor)
    {
        return {a.x * factor + b.x * (1 - factor), a.y * factor + b.y * (1 - factor), a.z * factor + b.z * (1 - factor)};
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

    __device__ __inline__ static float2 mix(float2 const& a, float2 const& b, float2 const& c, float factor1, float factor2)
    {
        float weight1 = factor1 * factor2;
        float weight2 = 1 - factor1;
        float weight3 = 1 - factor2;
        float sum = weight1 + weight2 + weight3;
        weight1 /= sum;
        weight2 /= sum;
        weight3 /= sum;
        return {a.x * weight1 + b.x * weight2 + c.x * weight3, a.y * weight1 + b.y * weight2 + c.y * weight3};
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
        return {a.x * weight1 + b.x * weight2 + c.x * weight3, a.y * weight1 + b.y * weight2 + c.y * weight3, a.z * weight1 + b.z * weight2 + c.z * weight3};
    }
};