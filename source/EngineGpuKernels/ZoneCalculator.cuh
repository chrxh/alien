#pragma once

#include "cuda_runtime_api.h"

#include "ConstantMemory.cuh"
#include "Util.cuh"
#include "Math.cuh"
#include "Map.cuh"
#include "SimulationData.cuh"

class ZoneCalculator
{
public:
    __device__ __inline__ static float calcParameterNew(BaseZoneParameter<float> const& baseZoneParameter, SimulationData const& data, float2 const& worldPos);
    __device__ __inline__ static float calcParameterNew(BaseZoneParameter<ColorVector<float>> const& baseZoneParameter, SimulationData const& data, float2 const& worldPos, int color);
    __device__ __inline__ static float calcParameterNew(BaseZoneParameter<ColorMatrix<float>> const& baseZoneParameter, SimulationData const& data, float2 const& worldPos, int color1, int color2);

    template <typename T, typename Parameter>
    __device__ __inline__ static T calcResultingValueNew(
        BaseMap const& map,
        float2 const& worldPos,
        T const& baseValue,
        T (&zoneValues)[MAX_ZONES],
        Parameter const& baseZoneParameter);

    //return -1 for base
    template <typename T>
    __device__ __inline__ static int
    getFirstMatchingZoneOrBaseNew(SimulationData const& data, float2 const& worldPos, BaseZoneParameter<T> const& parameter);

    __device__ __inline__ static bool isCoveredByZonesNew(SimulationData const& data, float2 const& worldPos, ZoneParameter<bool> const& enabledParameter);

    template <typename T>
    __device__ __inline__ static T calcResultingValue(BaseMap const& map, float2 const& worldPos, T const& baseValue, T (&spotValues)[MAX_ZONES]);

    template <typename T>
    __device__ __inline__ static T calcResultingFlowField(BaseMap const& map, float2 const& worldPos, T const& baseValue, T (&spotValues)[MAX_ZONES]);

private:

    __device__ __inline__ static float calcWeight(float2 const& delta, int const& zoneIndex);
    __device__ __inline__ static float calcWeightForCircularSpot(float2 const& delta, int const& spotIndex);
    __device__ __inline__ static float calcWeightForRectSpot(float2 const& delta, int const& spotIndex);

    template<typename T>
    __device__ __inline__ static T mix(T const& baseValue, T (&spotValues)[MAX_ZONES], float (&spotWeights)[MAX_ZONES], int numValues);

    template <typename T>
    __device__ __inline__ static T mix(T const& baseValue, T (&spotValues)[MAX_ZONES], float (&spotWeights)[MAX_ZONES]);
};


/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__device__ __inline__ float ZoneCalculator::calcParameterNew(BaseZoneParameter<float> const& baseZoneParameter, SimulationData const& data, float2 const& worldPos)
{
    float zoneValues[MAX_ZONES];
    int numValues = 0;
    for (int i = 0; i < cudaSimulationParameters.numZones.value; ++i) {
        if (baseZoneParameter.zoneValues[i].enabled) {
            zoneValues[numValues++] = baseZoneParameter.zoneValues[i].value;
        }
    }

    return calcResultingValueNew(data.cellMap, worldPos, baseZoneParameter.baseValue, zoneValues, baseZoneParameter);
}

__device__ __inline__ float
ZoneCalculator::calcParameterNew(BaseZoneParameter<ColorVector<float>> const& baseZoneParameter, SimulationData const& data, float2 const& worldPos, int color)
{
    float zoneValues[MAX_ZONES];
    int numValues = 0;
    for (int i = 0; i < cudaSimulationParameters.numZones.value; ++i) {
        if (baseZoneParameter.zoneValues[i].enabled) {
            zoneValues[numValues++] = baseZoneParameter.zoneValues[i].value[color];
        }
    }

    return calcResultingValueNew(data.cellMap, worldPos, baseZoneParameter.baseValue[color], zoneValues, baseZoneParameter);
}

__device__ __inline__ float ZoneCalculator::calcParameterNew(
    BaseZoneParameter<ColorMatrix<float>> const& baseZoneParameter,
    SimulationData const& data,
    float2 const& worldPos,
    int color1,
    int color2)
{
    float zoneValues[MAX_ZONES];
    int numValues = 0;
    for (int i = 0; i < cudaSimulationParameters.numZones.value; ++i) {
        if (baseZoneParameter.zoneValues[i].enabled) {
            zoneValues[numValues++] = baseZoneParameter.zoneValues[i].value[color1][color2];
        }
    }

    return calcResultingValueNew(data.cellMap, worldPos, baseZoneParameter.baseValue[color1][color2], zoneValues, baseZoneParameter);
}

template<typename T>
__device__ __inline__ int
ZoneCalculator::getFirstMatchingZoneOrBaseNew(SimulationData const& data, float2 const& worldPos, BaseZoneParameter<T> const& parameter)
{
    auto const& map = data.cellMap;
    for (int i = 0; i < cudaSimulationParameters.numZones.value; ++i) {
        if (parameter.zoneValues[i].enabled) {
            float2 spotPos = {cudaSimulationParameters.zone[i].posX, cudaSimulationParameters.zone[i].posY};
            auto delta = map.getCorrectedDirection(spotPos - worldPos);
            if (calcWeight(delta, i) < NEAR_ZERO) {
                return i;
            }
        }
    }
    return -1;
}

__device__ __inline__ bool ZoneCalculator::isCoveredByZonesNew(SimulationData const& data, float2 const& worldPos, ZoneParameter<bool> const& enabledParameter)
{
    auto const& map = data.cellMap;
    for (int i = 0; i < cudaSimulationParameters.numZones.value; ++i) {
        if (enabledParameter.zoneValues[i]) {
            float2 spotPos = {cudaSimulationParameters.zone[i].posX, cudaSimulationParameters.zone[i].posY};
            auto delta = map.getCorrectedDirection(spotPos - worldPos);
            if (calcWeight(delta, i) < NEAR_ZERO) {
                return true;
            }
        }
    }
    return false;
}

template <typename T, typename Parameter>
__device__ __inline__ T ZoneCalculator::calcResultingValueNew(
    BaseMap const& map,
    float2 const& worldPos,
    T const& baseValue,
    T (&zoneValues)[MAX_ZONES],
    Parameter const& baseZoneParameter)
{
    if (0 == cudaSimulationParameters.numZones.value) {
        return baseValue;
    } else {
        float spotWeights[MAX_ZONES];
        int numValues = 0;
        for (int i = 0; i < cudaSimulationParameters.numZones.value; ++i) {
            if (baseZoneParameter.zoneValues[i].enabled) {
                float2 spotPos = {cudaSimulationParameters.zone[i].posX, cudaSimulationParameters.zone[i].posY};
                auto delta = map.getCorrectedDirection(spotPos - worldPos);
                spotWeights[numValues++] = calcWeight(delta, i);
            }
        }
        return mix(baseValue, zoneValues, spotWeights, numValues);
    }
}

template <typename T>
__device__ __inline__ T ZoneCalculator::calcResultingValue(BaseMap const& map, float2 const& worldPos, T const& baseValue, T (&spotValues)[MAX_ZONES])
{
    if (0 == cudaSimulationParameters.numZones.value) {
        return baseValue;
    } else {
        float spotWeights[MAX_ZONES];
        for (int i = 0; i < cudaSimulationParameters.numZones.value; ++i) {
            float2 spotPos = {cudaSimulationParameters.zone[i].posX, cudaSimulationParameters.zone[i].posY};
            auto delta = map.getCorrectedDirection(spotPos - worldPos);
            spotWeights[i] = calcWeight(delta, i);
        }
        return mix(baseValue, spotValues, spotWeights);
    }
}

template <typename T>
__device__ __inline__ T ZoneCalculator::calcResultingFlowField(BaseMap const& map, float2 const& worldPos, T const& baseValue, T (&spotValues)[MAX_ZONES])
{
    if (0 == cudaSimulationParameters.numZones.value) {
        return baseValue;
    } else {
        float spotWeights[MAX_ZONES];
        int numValues = 0;
        for (int i = 0; i < cudaSimulationParameters.numZones.value; ++i) {
            if (cudaSimulationParameters.zone[i].flow.type != FlowType_None) {
                float2 spotPos = {cudaSimulationParameters.zone[i].posX, cudaSimulationParameters.zone[i].posY};
                auto delta = map.getCorrectedDirection(spotPos - worldPos);
                spotWeights[numValues++] = calcWeight(delta, i);
            }
        }
        return mix(baseValue, spotValues, spotWeights, numValues);
    }
}

__device__ __inline__ float ZoneCalculator::calcWeight(float2 const& delta, int const& zoneIndex)
{
    if (cudaSimulationParameters.zone[zoneIndex].shape.type == ZoneShapeType_Rectangular) {
        return calcWeightForRectSpot(delta, zoneIndex);
    } else {
        return calcWeightForCircularSpot(delta, zoneIndex);
    }
}

__device__ __inline__ float ZoneCalculator::calcWeightForCircularSpot(float2 const& delta, int const& spotIndex)
{
    auto distance = Math::length(delta);
    auto coreRadius = cudaSimulationParameters.zone[spotIndex].shape.alternatives.circularSpot.coreRadius;
    auto fadeoutRadius = cudaSimulationParameters.zone[spotIndex].fadeoutRadius + 1;
    return distance < coreRadius ? 0.0f : min(1.0f, (distance - coreRadius) / fadeoutRadius);
}

__device__ __inline__ float ZoneCalculator::calcWeightForRectSpot(float2 const& delta, int const& spotIndex)
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

template <typename T>
__device__ __inline__ T ZoneCalculator::mix(T const& baseValue, T (&spotValues)[MAX_ZONES], float (&spotWeights)[MAX_ZONES], int numValues)
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
__device__ __inline__ T ZoneCalculator::mix(T const& baseValue, T (&spotValues)[MAX_ZONES], float (&spotWeights)[MAX_ZONES])
{
    float baseFactor = 1;
    float sum = 0;
    for (int i = 0; i < cudaSimulationParameters.numZones.value; ++i) {
        baseFactor *= spotWeights[i];
        sum += 1.0f - spotWeights[i];
    }
    sum += baseFactor;
    T result = baseValue * baseFactor;
    for (int i = 0; i < cudaSimulationParameters.numZones.value; ++i) {
        result += spotValues[i] * (1.0f - spotWeights[i]) / sum;
    }
    return result;
}
