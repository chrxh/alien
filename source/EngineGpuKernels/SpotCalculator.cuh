#pragma once

#include "cuda_runtime_api.h"

#include "EngineInterface/SimulationParametersSpotValues.h"
#include "ConstantMemory.cuh"
#include "Swap.cuh"
#include "Math.cuh"

class SpotCalculator
{
public:
    template <typename T>
    __device__ __inline__ static T calcResultingValue(
        BaseMap const& map,
        float2 const& worldPos,
        T const& baseValue,
        T (&spotValues)[MAX_SPOTS],
        bool SimulationParametersSpotActivatedValues::*valueActivated)
    {
        if (0 == cudaSimulationParameters.numSpots) {
            return baseValue;
        } else {
            float spotWeights[MAX_SPOTS];
            int numValues = 0;
            for (int i = 0; i < cudaSimulationParameters.numSpots; ++i) {
                if (cudaSimulationParameters.spots[i].activatedValues.*valueActivated) {
                    float2 spotPos = {cudaSimulationParameters.spots[i].posX, cudaSimulationParameters.spots[i].posY};
                    auto delta = map.getCorrectedDirection(spotPos - worldPos);
                    spotWeights[numValues++] = calcWeight(delta, i);
                }
            }
            return mix(baseValue, spotValues, spotWeights, numValues);
        }
    }

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

    template <typename T>
    __device__ __inline__ static T calcResultingFlowField(
        BaseMap const& map,
        float2 const& worldPos,
        T const& baseValue,
        T (&spotValues)[MAX_SPOTS])
    {
        if (0 == cudaSimulationParameters.numSpots) {
            return baseValue;
        } else {
            float spotWeights[MAX_SPOTS];
            int numValues = 0;
            for (int i = 0; i < cudaSimulationParameters.numSpots; ++i) {
                if (cudaSimulationParameters.spots[i].flowType != FlowType_None) {
                    float2 spotPos = {cudaSimulationParameters.spots[i].posX, cudaSimulationParameters.spots[i].posY};
                    auto delta = map.getCorrectedDirection(spotPos - worldPos);
                    spotWeights[numValues++] = calcWeight(delta, i);
                }
            }
            return mix(baseValue, spotValues, spotWeights, numValues);
        }
    }

    __device__ __inline__ static float calcParameter(
        float SimulationParametersSpotValues::*value,
        bool SimulationParametersSpotActivatedValues::*valueActivated,
        SimulationData const& data,
        float2 const& worldPos)
    {
        float spotValues[MAX_SPOTS];
        int numValues = 0;
        for (int i = 0; i < cudaSimulationParameters.numSpots; ++i) {
            if (cudaSimulationParameters.spots[i].activatedValues.*valueActivated) {
                spotValues[numValues++] = cudaSimulationParameters.spots[i].values.*value;
            }
        }

        return calcResultingValue(data.cellMap, worldPos, cudaSimulationParameters.baseValues.*value, spotValues, valueActivated);
    }

    __device__ __inline__ static float calcParameter(
        ColorVector<float> SimulationParametersSpotValues::*value,
        bool SimulationParametersSpotActivatedValues::*valueActivated,
        SimulationData const& data,
        float2 const& worldPos,
        int color)
    {
        float spotValues[MAX_SPOTS];
        int numValues = 0;
        for (int i = 0; i < cudaSimulationParameters.numSpots; ++i) {
            if (cudaSimulationParameters.spots[i].activatedValues.*valueActivated) {
                spotValues[numValues++] = (cudaSimulationParameters.spots[i].values.*value)[color];
            }
        }

        return calcResultingValue(data.cellMap, worldPos, (cudaSimulationParameters.baseValues.*value)[color], spotValues, valueActivated);
    }

    __device__ __inline__ static int calcParameter(
        int SimulationParametersSpotValues::*value,
        bool SimulationParametersSpotActivatedValues::*valueActivated,
        SimulationData const& data,
        float2 const& worldPos)
    {
        float spotValues[MAX_SPOTS];
        int numValues = 0;
        for (int i = 0; i < cudaSimulationParameters.numSpots; ++i) {
            if (cudaSimulationParameters.spots[i].activatedValues.*valueActivated) {
                spotValues[numValues++] = toFloat(cudaSimulationParameters.spots[i].values.*value);
            }
        }

        return toInt(calcResultingValue(data.cellMap, worldPos, toFloat(cudaSimulationParameters.baseValues.*value), spotValues, valueActivated));
    }

    __device__ __inline__ static float calcParameter(
        ColorMatrix<float> SimulationParametersSpotValues::*value,
        bool SimulationParametersSpotActivatedValues::*valueActivated,
        SimulationData const& data,
        float2 const& worldPos,
        int color1,
        int color2)
    {
        float spotValues[MAX_SPOTS];
        int numValues = 0;
        for (int i = 0; i < cudaSimulationParameters.numSpots; ++i) {
            if (cudaSimulationParameters.spots[i].activatedValues.*valueActivated) {
                spotValues[numValues++] = (cudaSimulationParameters.spots[i].values.*value)[color1][color2];
            }
        }

        return calcResultingValue(data.cellMap, worldPos, (cudaSimulationParameters.baseValues.*value)[color1][color2], spotValues, valueActivated);
    }

    //return -1 for base
    __device__ __inline__ static int
    getFirstMatchingSpotOrBase(SimulationData const& data, float2 const& worldPos, bool SimulationParametersSpotActivatedValues::*valueActivated)
    {
        if (0 == cudaSimulationParameters.numSpots) {
            return -1;
        } else {
            auto const& map = data.cellMap;
            for (int i = 0; i < cudaSimulationParameters.numSpots; ++i) {
                if (cudaSimulationParameters.spots[i].activatedValues.*valueActivated) {
                    float2 spotPos = {cudaSimulationParameters.spots[i].posX, cudaSimulationParameters.spots[i].posY};
                    auto delta = map.getCorrectedDirection(spotPos - worldPos);
                    if(calcWeight(delta, i) < NEAR_ZERO) {
                        return i;
                    }
                }
            }
            return -1;
        }
    }

private:

    __device__ __inline__ static float calcWeight(float2 const& delta, int const& spotIndex)
    {
        if (cudaSimulationParameters.spots[spotIndex].shapeType == SpotShapeType_Rectangular) {
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

    template<typename T>
    __device__ __inline__ static T mix(T const& baseValue, T (&spotValues)[MAX_SPOTS], float (&spotWeights)[MAX_SPOTS], int numValues)
    {
        float baseFactor = 1;
        float sum = 0;
        for (int i = 0; i < numValues; ++i) {
            baseFactor *= spotWeights[i];
            sum += 1.0f - spotWeights[i];
        }
        sum += baseFactor;
        T result = baseValue * baseFactor;
        for (int i = 0; i < numValues; ++i) {
            result += spotValues[i] * (1.0f - spotWeights[i]) / sum;
        }
        return result;
    }

    template <typename T>
    __device__ __inline__ static T mix(T const& baseValue, T (&spotValues)[MAX_SPOTS], float (&spotWeights)[MAX_SPOTS])
    {
        float baseFactor = 1;
        float sum = 0;
        for (int i = 0; i < cudaSimulationParameters.numSpots; ++i) {
            baseFactor *= spotWeights[i];
            sum += 1.0f - spotWeights[i];
        }
        sum += baseFactor;
        T result = baseValue * baseFactor;
        for (int i = 0; i < cudaSimulationParameters.numSpots; ++i) {
            result += spotValues[i] * (1.0f - spotWeights[i]) / sum;
        }
        return result;
    }

};