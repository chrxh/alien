#pragma once

#include "cuda_runtime_api.h"

#include "EngineInterface/SimulationParametersSpotValues.h"
#include "ConstantMemory.cuh"
#include "Swap.cuh"

class SpotCalculator
{
public:
    template <typename T>
    __device__ __inline__ static T calcResultingValue(BaseMap const& map, float2 const& worldPos, T const& baseValue, T (&spotValues)[MAX_SPOTS])
    {
        if (0 == cudaSimulationParameters.numSpots) {
            return baseValue;
        } else {
            float spotWeights[MAX_SPOTS];
            for (int i = 0; i < cudaSimulationParameters.numSpots; ++i) {
                float2 spotPos = {cudaSimulationParameters.spots[i].posX, cudaSimulationParameters.spots[i].posY};
                auto delta = map.getCorrectedDirection(spotPos - worldPos);
                spotWeights[i] = calcWeight(delta, i);
            }
            return mix(baseValue, spotValues, spotWeights);
        }
    }

    __device__ __inline__ static float calcParameter(float SimulationParametersSpotValues::*value, SimulationData const& data, float2 const& worldPos)
    {
        float spotValues[MAX_SPOTS];
        for (int i = 0; i < cudaSimulationParameters.numSpots; ++i) {
            spotValues[i] = cudaSimulationParameters.spots[i].values.*value;
        }

        return calcResultingValue(data.cellMap, worldPos, cudaSimulationParameters.baseValues.*value, spotValues);
    }

    __device__ __inline__ static int calcParameter(int SimulationParametersSpotValues::*value, SimulationData const& data, float2 const& worldPos)
    {
        float spotValues[MAX_SPOTS];
        for (int i = 0; i < cudaSimulationParameters.numSpots; ++i) {
            spotValues[i] = toFloat(cudaSimulationParameters.spots[i].values.*value);
        }

        return toInt(calcResultingValue(data.cellMap, worldPos, toFloat(cudaSimulationParameters.baseValues.*value), spotValues));
    }

    __device__ __inline__ static float calcColorMatrix(int color, int otherColor, SimulationData const& data, float2 const& worldPos)
    {
        float spotValues[MAX_SPOTS];
        for (int i = 0; i < cudaSimulationParameters.numSpots; ++i) {
            spotValues[i] = cudaSimulationParameters.spots[i].values.cellFunctionAttackerFoodChainColorMatrix[color][otherColor];
        }

        return calcResultingValue(
            data.cellMap, worldPos, cudaSimulationParameters.baseValues.cellFunctionAttackerFoodChainColorMatrix[color][otherColor], spotValues);
    }

    __device__ __inline__ static int calcColorTransitionDuration(int color, SimulationData const& data, float2 const& worldPos)
    {
        float spotValues[MAX_SPOTS];
        for (int i = 0; i < cudaSimulationParameters.numSpots; ++i) {
            spotValues[i] = toFloat(cudaSimulationParameters.spots[i].values.cellColorTransitionDuration[color]);
        }

        return toInt(calcResultingValue(data.cellMap, worldPos, toFloat(cudaSimulationParameters.baseValues.cellColorTransitionDuration[color]), spotValues));
    }

    __device__ __inline__ static int calcColorTransitionTargetColor(int color, SimulationData const& data, float2 const& worldPos)
    {
        float spotValues[MAX_SPOTS];
        for (int i = 0; i < cudaSimulationParameters.numSpots; ++i) {
            spotValues[i] = toFloat(cudaSimulationParameters.spots[i].values.cellColorTransitionTargetColor[color]);
        }

        return toInt(
            calcResultingValue(data.cellMap, worldPos, toFloat(cudaSimulationParameters.baseValues.cellColorTransitionTargetColor[color]), spotValues) + 0.5f);
    }

private:

    __device__ __inline__ static float calcWeight(float2 const& delta, int const& spotIndex)
    {
        if (cudaSimulationParameters.spots[spotIndex].shapeType == ShapeType_Rectangular) {
            return calcWeightForRectSpot(delta, spotIndex);
        } else {
            return calcWeightForCircularSpot(delta, spotIndex);
        }
    }

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

        __device__ __inline__ static float mix(float const& baseValue, float (&spotValues)[MAX_SPOTS], float (&spotWeights)[MAX_SPOTS])
    {
        float baseFactor = 1;
        float sum = 0;
        for (int i = 0; i < cudaSimulationParameters.numSpots; ++i) {
            baseFactor *= spotWeights[i];
            sum += 1 - spotWeights[i];
        }
        sum += baseFactor;
        float result = baseValue * baseFactor;
        for (int i = 0; i < cudaSimulationParameters.numSpots; ++i) {
            result += spotValues[i] * (1 - spotWeights[i]) / sum;
        }
        return result;
    }

    __device__ __inline__ static float2 mix(float2 const& baseValue, float2 (&spotValues)[MAX_SPOTS], float (&spotWeights)[MAX_SPOTS])
    {
        float baseFactor = 1;
        float sum = 0;
        for (int i = 0; i < cudaSimulationParameters.numSpots; ++i) {
            baseFactor *= spotWeights[i];
            sum += 1 - spotWeights[i];
        }
        sum += baseFactor;
        float2 result = {baseValue.x * baseFactor, baseValue.y * baseFactor};
        for (int i = 0; i < cudaSimulationParameters.numSpots; ++i) {
            result.x += spotValues[i].x * (1 - spotWeights[i]) / sum;
            result.y += spotValues[i].y * (1 - spotWeights[i]) / sum;
        }
        return result;
    }

    __device__ __inline__ static float3 mix(float3 const& baseValue, float3 (&spotValues)[MAX_SPOTS], float (&spotWeights)[MAX_SPOTS])
    {
        float baseFactor = 1;
        float sum = 0;
        for (int i = 0; i < cudaSimulationParameters.numSpots; ++i) {
            baseFactor *= spotWeights[i];
            sum += 1 - spotWeights[i];
        }
        sum += baseFactor;
        float3 result = {baseValue.x * baseFactor, baseValue.y * baseFactor, baseValue.z * baseFactor};
        for (int i = 0; i < cudaSimulationParameters.numSpots; ++i) {
            result.x += spotValues[i].x * (1 - spotWeights[i]) / sum;
            result.y += spotValues[i].y * (1 - spotWeights[i]) / sum;
            result.z += spotValues[i].z * (1 - spotWeights[i]) / sum;
        }
        return result;
    }
};