#pragma once

#include "cuda_runtime_api.h"

#include "EngineInterface/SimulationParametersZoneValues.h"
#include "ConstantMemory.cuh"
#include "Util.cuh"
#include "Math.cuh"
#include "Map.cuh"
#include "SimulationData.cuh"

class ZoneCalculator
{
public:
    //NEW
    __device__ __inline__ static float calcParameterNew(BaseZoneParameter<float> const& baseZoneParameter, SimulationData const& data, float2 const& worldPos);

    template <typename T>
    __device__ __inline__ static T calcResultingValueNew(
        BaseMap const& map,
        float2 const& worldPos,
        T const& baseValue,
        T (&spotValues)[MAX_ZONES],
        BaseZoneParameter<float> const& baseZoneParameter);

    //OLD
    template <typename T>
    __device__ __inline__ static T calcResultingValue(
        BaseMap const& map,
        float2 const& worldPos,
        T const& baseValue,
        T (&spotValues)[MAX_ZONES],
        bool SimulationParametersZoneEnabledValues::*valueActivated)
    {
        if (0 == cudaSimulationParameters.numZones) {
            return baseValue;
        } else {
            float spotWeights[MAX_ZONES];
            int numValues = 0;
            for (int i = 0; i < cudaSimulationParameters.numZones; ++i) {
                if (cudaSimulationParameters.zone[i].enabledValues.*valueActivated) {
                    float2 spotPos = {cudaSimulationParameters.zone[i].posX, cudaSimulationParameters.zone[i].posY};
                    auto delta = map.getCorrectedDirection(spotPos - worldPos);
                    spotWeights[numValues++] = calcWeight(delta, i);
                }
            }
            return mix(baseValue, spotValues, spotWeights, numValues);
        }
    }

    template <typename T>
    __device__ __inline__ static T calcResultingValue(BaseMap const& map, float2 const& worldPos, T const& baseValue, T (&spotValues)[MAX_ZONES])
    {
        if (0 == cudaSimulationParameters.numZones) {
            return baseValue;
        } else {
            float spotWeights[MAX_ZONES];
            for (int i = 0; i < cudaSimulationParameters.numZones; ++i) {
                float2 spotPos = {cudaSimulationParameters.zone[i].posX, cudaSimulationParameters.zone[i].posY};
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
        T (&spotValues)[MAX_ZONES])
    {
        if (0 == cudaSimulationParameters.numZones) {
            return baseValue;
        } else {
            float spotWeights[MAX_ZONES];
            int numValues = 0;
            for (int i = 0; i < cudaSimulationParameters.numZones; ++i) {
                if (cudaSimulationParameters.zone[i].flow.type != FlowType_None) {
                    float2 spotPos = {cudaSimulationParameters.zone[i].posX, cudaSimulationParameters.zone[i].posY};
                    auto delta = map.getCorrectedDirection(spotPos - worldPos);
                    spotWeights[numValues++] = calcWeight(delta, i);
                }
            }
            return mix(baseValue, spotValues, spotWeights, numValues);
        }
    }

    __device__ __inline__ static float calcParameter(
        float SimulationParametersZoneValues::*value,
        bool SimulationParametersZoneEnabledValues::*valueActivated,
        SimulationData const& data,
        float2 const& worldPos)
    {
        float spotValues[MAX_ZONES];
        int numValues = 0;
        for (int i = 0; i < cudaSimulationParameters.numZones; ++i) {
            if (cudaSimulationParameters.zone[i].enabledValues.*valueActivated) {
                spotValues[numValues++] = cudaSimulationParameters.zone[i].values.*value;
            }
        }

        return calcResultingValue(data.cellMap, worldPos, cudaSimulationParameters.baseValues.*value, spotValues, valueActivated);
    }

    __device__ __inline__ static float calcParameter(
        ColorVector<float> SimulationParametersZoneValues::*value,
        bool SimulationParametersZoneEnabledValues::*valueActivated,
        SimulationData const& data,
        float2 const& worldPos,
        int color)
    {
        float spotValues[MAX_ZONES];
        int numValues = 0;
        for (int i = 0; i < cudaSimulationParameters.numZones; ++i) {
            if (cudaSimulationParameters.zone[i].enabledValues.*valueActivated) {
                spotValues[numValues++] = (cudaSimulationParameters.zone[i].values.*value)[color];
            }
        }

        return calcResultingValue(data.cellMap, worldPos, (cudaSimulationParameters.baseValues.*value)[color], spotValues, valueActivated);
    }

    __device__ __inline__ static int calcParameter(
        int SimulationParametersZoneValues::*value,
        bool SimulationParametersZoneEnabledValues::*valueActivated,
        SimulationData const& data,
        float2 const& worldPos)
    {
        float spotValues[MAX_ZONES];
        int numValues = 0;
        for (int i = 0; i < cudaSimulationParameters.numZones; ++i) {
            if (cudaSimulationParameters.zone[i].enabledValues.*valueActivated) {
                spotValues[numValues++] = toFloat(cudaSimulationParameters.zone[i].values.*value);
            }
        }

        return toInt(calcResultingValue(data.cellMap, worldPos, toFloat(cudaSimulationParameters.baseValues.*value), spotValues, valueActivated));
    }

    __device__ __inline__ static bool calcParameter(
        bool SimulationParametersZoneValues::*value,
        bool SimulationParametersZoneEnabledValues::*valueActivated,
        SimulationData const& data,
        float2 const& worldPos)
    {
        float spotValues[MAX_ZONES];
        int numValues = 0;
        for (int i = 0; i < cudaSimulationParameters.numZones; ++i) {
            if (cudaSimulationParameters.zone[i].enabledValues.*valueActivated) {
                spotValues[numValues++] = cudaSimulationParameters.zone[i].values.*value ? 1.0f : 0.0f;
            }
        }
        return calcResultingValue(data.cellMap, worldPos, cudaSimulationParameters.baseValues.*value ? 1.0f : 0.0f, spotValues, valueActivated) > 0.5f;
    }

    __device__ __inline__ static float calcParameter(
        ColorMatrix<float> SimulationParametersZoneValues::*value,
        bool SimulationParametersZoneEnabledValues::*valueActivated,
        SimulationData const& data,
        float2 const& worldPos,
        int color1,
        int color2)
    {
        float spotValues[MAX_ZONES];
        int numValues = 0;
        for (int i = 0; i < cudaSimulationParameters.numZones; ++i) {
            if (cudaSimulationParameters.zone[i].enabledValues.*valueActivated) {
                spotValues[numValues++] = (cudaSimulationParameters.zone[i].values.*value)[color1][color2];
            }
        }

        return calcResultingValue(data.cellMap, worldPos, (cudaSimulationParameters.baseValues.*value)[color1][color2], spotValues, valueActivated);
    }

    //return -1 for base
    __device__ __inline__ static int
    getFirstMatchingZoneOrBase(SimulationData const& data, float2 const& worldPos, bool SimulationParametersZoneEnabledValues::*valueActivated)
    {
        if (0 == cudaSimulationParameters.numZones) {
            return -1;
        } else {
            auto const& map = data.cellMap;
            for (int i = 0; i < cudaSimulationParameters.numZones; ++i) {
                if (cudaSimulationParameters.zone[i].enabledValues.*valueActivated) {
                    float2 spotPos = {cudaSimulationParameters.zone[i].posX, cudaSimulationParameters.zone[i].posY};
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

    __device__ __inline__ static float calcWeight(float2 const& delta, int const& zoneIndex)
    {
        if (cudaSimulationParameters.zone[zoneIndex].shape.type == ZoneShapeType_Rectangular) {
            return calcWeightForRectSpot(delta, zoneIndex);
        } else {
            return calcWeightForCircularSpot(delta, zoneIndex);
        }
    }

    __device__ __inline__ static float calcWeightForCircularSpot(float2 const& delta, int const& spotIndex)
    {
        auto distance = Math::length(delta);
        auto coreRadius = cudaSimulationParameters.zone[spotIndex].shape.alternatives.circularSpot.coreRadius;
        auto fadeoutRadius = cudaSimulationParameters.zone[spotIndex].fadeoutRadius + 1;
        return distance < coreRadius ? 0.0f : min(1.0f, (distance - coreRadius) / fadeoutRadius);
    }

    __device__ __inline__ static float calcWeightForRectSpot(float2 const& delta, int const& spotIndex)
    {
        auto const& spot = cudaSimulationParameters.zone[spotIndex];
        float result = 0;
        if (abs(delta.x) > spot.shape.alternatives.rectangularSpot.width / 2 || abs(delta.y) > spot.shape.alternatives.rectangularSpot.height / 2) {
            float2 distanceFromRect = {
                max(0.0f, abs(delta.x) - spot.shape.alternatives.rectangularSpot.width / 2),
                max(0.0f, abs(delta.y) - spot.shape.alternatives.rectangularSpot.height / 2)};
            result = min(1.0f, Math::length(distanceFromRect) / (spot.fadeoutRadius + 1));
        }
        return result;
    }

    template<typename T>
    __device__ __inline__ static T mix(T const& baseValue, T (&spotValues)[MAX_ZONES], float (&spotWeights)[MAX_ZONES], int numValues)
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
    __device__ __inline__ static T mix(T const& baseValue, T (&spotValues)[MAX_ZONES], float (&spotWeights)[MAX_ZONES])
    {
        float baseFactor = 1;
        float sum = 0;
        for (int i = 0; i < cudaSimulationParameters.numZones; ++i) {
            baseFactor *= spotWeights[i];
            sum += 1.0f - spotWeights[i];
        }
        sum += baseFactor;
        T result = baseValue * baseFactor;
        for (int i = 0; i < cudaSimulationParameters.numZones; ++i) {
            result += spotValues[i] * (1.0f - spotWeights[i]) / sum;
        }
        return result;
    }

};


/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__device__ __inline__ float ZoneCalculator::calcParameterNew(BaseZoneParameter<float> const& baseZoneParameter, SimulationData const& data, float2 const& worldPos)
{
    float zoneValues[MAX_ZONES];
    int numValues = 0;
    for (int i = 0; i < cudaSimulationParameters.numZones; ++i) {
        if (baseZoneParameter.zoneValues[i].enabled) {
            zoneValues[numValues++] = baseZoneParameter.zoneValues[i].value;
        }
    }

    return calcResultingValueNew(data.cellMap, worldPos, baseZoneParameter.baseValue, zoneValues, baseZoneParameter);
}

template <typename T>
__device__ __inline__ T ZoneCalculator::calcResultingValueNew(
    BaseMap const& map,
    float2 const& worldPos,
    T const& baseValue,
    T (&spotValues)[MAX_ZONES],
    BaseZoneParameter<float> const& baseZoneParameter)
{
    if (0 == cudaSimulationParameters.numZones) {
        return baseValue;
    } else {
        float spotWeights[MAX_ZONES];
        int numValues = 0;
        for (int i = 0; i < cudaSimulationParameters.numZones; ++i) {
            if (baseZoneParameter.zoneValues[i].enabled) {
                float2 spotPos = {cudaSimulationParameters.zone[i].posX, cudaSimulationParameters.zone[i].posY};
                auto delta = map.getCorrectedDirection(spotPos - worldPos);
                spotWeights[numValues++] = calcWeight(delta, i);
            }
        }
        return mix(baseValue, spotValues, spotWeights, numValues);
    }
}
