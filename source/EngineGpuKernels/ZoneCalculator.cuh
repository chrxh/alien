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
    __device__ __inline__ static T calcResultingValue(BaseMap const& map, float2 const& worldPos, T const& baseValue, T (&zoneValues)[MAX_ZONES]);

    template <typename T>
    __device__ __inline__ static T calcResultingFlowField(BaseMap const& map, float2 const& worldPos, T const& baseValue, T (&zoneValues)[MAX_ZONES]);

private:

    __device__ __inline__ static float calcWeight(float2 const& delta, int const& zoneIndex);
    __device__ __inline__ static float calcWeightForCircularZone(float2 const& delta, int const& zoneIndex);
    __device__ __inline__ static float calcWeightForRectZone(float2 const& delta, int const& zoneIndex);

    template<typename T>
    __device__ __inline__ static T mix(T const& baseValue, T (&zoneValues)[MAX_ZONES], float (&zoneWeights)[MAX_ZONES], int numValues);

    template <typename T>
    __device__ __inline__ static T mix(T const& baseValue, T (&zoneValues)[MAX_ZONES], float (&zoneWeights)[MAX_ZONES]);
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
            float2 zonePos = {cudaSimulationParameters.zonePosition.zoneValues[i].x, cudaSimulationParameters.zonePosition.zoneValues[i].y};
            auto delta = map.getCorrectedDirection(zonePos - worldPos);
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
            float2 zonePos = {cudaSimulationParameters.zonePosition.zoneValues[i].x, cudaSimulationParameters.zonePosition.zoneValues[i].y};
            auto delta = map.getCorrectedDirection(zonePos - worldPos);
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
        float zoneWeights[MAX_ZONES];
        int numValues = 0;
        for (int i = 0; i < cudaSimulationParameters.numZones.value; ++i) {
            if (baseZoneParameter.zoneValues[i].enabled) {
                float2 zonePos = {cudaSimulationParameters.zonePosition.zoneValues[i].x, cudaSimulationParameters.zonePosition.zoneValues[i].y};
                auto delta = map.getCorrectedDirection(zonePos - worldPos);
                zoneWeights[numValues++] = calcWeight(delta, i);
            }
        }
        return mix(baseValue, zoneValues, zoneWeights, numValues);
    }
}

template <typename T>
__device__ __inline__ T ZoneCalculator::calcResultingValue(BaseMap const& map, float2 const& worldPos, T const& baseValue, T (&zoneValues)[MAX_ZONES])
{
    if (0 == cudaSimulationParameters.numZones.value) {
        return baseValue;
    } else {
        float zoneWeights[MAX_ZONES];
        for (int i = 0; i < cudaSimulationParameters.numZones.value; ++i) {
            float2 zonePos = {cudaSimulationParameters.zonePosition.zoneValues[i].x, cudaSimulationParameters.zonePosition.zoneValues[i].y};
            auto delta = map.getCorrectedDirection(zonePos - worldPos);
            zoneWeights[i] = calcWeight(delta, i);
        }
        return mix(baseValue, zoneValues, zoneWeights);
    }
}

template <typename T>
__device__ __inline__ T ZoneCalculator::calcResultingFlowField(BaseMap const& map, float2 const& worldPos, T const& baseValue, T (&zoneValues)[MAX_ZONES])
{
    if (0 == cudaSimulationParameters.numZones.value) {
        return baseValue;
    } else {
        float zoneWeights[MAX_ZONES];
        int numValues = 0;
        for (int i = 0; i < cudaSimulationParameters.numZones.value; ++i) {
            if (cudaSimulationParameters.zone[i].flow.type != FlowType_None) {
                float2 zonePos = {cudaSimulationParameters.zonePosition.zoneValues[i].x, cudaSimulationParameters.zonePosition.zoneValues[i].y};
                auto delta = map.getCorrectedDirection(zonePos - worldPos);
                zoneWeights[numValues++] = calcWeight(delta, i);
            }
        }
        return mix(baseValue, zoneValues, zoneWeights, numValues);
    }
}

__device__ __inline__ float ZoneCalculator::calcWeight(float2 const& delta, int const& zoneIndex)
{
    if (cudaSimulationParameters.zoneShape.zoneValues[zoneIndex] == ZoneShapeType_Rectangular) {
        return calcWeightForRectZone(delta, zoneIndex);
    } else {
        return calcWeightForCircularZone(delta, zoneIndex);
    }
}

__device__ __inline__ float ZoneCalculator::calcWeightForCircularZone(float2 const& delta, int const& zoneIndex)
{
    auto distance = Math::length(delta);
    auto coreRadius = cudaSimulationParameters.zoneCoreRadius.zoneValues[zoneIndex];
    auto fadeoutRadius = cudaSimulationParameters.zone[zoneIndex].fadeoutRadius + 1;
    return distance < coreRadius ? 0.0f : min(1.0f, (distance - coreRadius) / fadeoutRadius);
}

__device__ __inline__ float ZoneCalculator::calcWeightForRectZone(float2 const& delta, int const& zoneIndex)
{
    auto const& zone = cudaSimulationParameters.zone[zoneIndex];
    float result = 0;
    if (abs(delta.x) > cudaSimulationParameters.zoneCoreRect.zoneValues[zoneIndex].x / 2
        || abs(delta.y) > cudaSimulationParameters.zoneCoreRect.zoneValues[zoneIndex].y / 2) {
        float2 distanceFromRect = {
            max(0.0f, abs(delta.x) - cudaSimulationParameters.zoneCoreRect.zoneValues[zoneIndex].x / 2),
            max(0.0f, abs(delta.y) - cudaSimulationParameters.zoneCoreRect.zoneValues[zoneIndex].y / 2)};
        result = min(1.0f, Math::length(distanceFromRect) / (zone.fadeoutRadius + 1));
    }
    return result;
}

template <typename T>
__device__ __inline__ T ZoneCalculator::mix(T const& baseValue, T (&zoneValues)[MAX_ZONES], float (&zoneWeights)[MAX_ZONES], int numValues)
{
    float baseFactor = 1;
    float sum = 0;
    for (int i = 0; i < numValues; ++i) {
        baseFactor *= zoneWeights[i];
        sum += 1.0f - zoneWeights[i];
    }
    sum += baseFactor;
    T result = baseValue * baseFactor;
    for (int i = 0; i < numValues; ++i) {
        result += zoneValues[i] * (1.0f - zoneWeights[i]) / sum;
    }
    return result;
}

template <typename T>
__device__ __inline__ T ZoneCalculator::mix(T const& baseValue, T (&zoneValues)[MAX_ZONES], float (&zoneWeights)[MAX_ZONES])
{
    float baseFactor = 1;
    float sum = 0;
    for (int i = 0; i < cudaSimulationParameters.numZones.value; ++i) {
        baseFactor *= zoneWeights[i];
        sum += 1.0f - zoneWeights[i];
    }
    sum += baseFactor;
    T result = baseValue * baseFactor;
    for (int i = 0; i < cudaSimulationParameters.numZones.value; ++i) {
        result += zoneValues[i] * (1.0f - zoneWeights[i]) / sum;
    }
    return result;
}
