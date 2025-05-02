#pragma once

#include "cuda_runtime_api.h"

#include "ConstantMemory.cuh"
#include "Util.cuh"
#include "Math.cuh"
#include "Map.cuh"
#include "SimulationData.cuh"

class ParameterCalculator
{
public:
    __device__ __inline__ static float calcParameter(BaseLayerParameter<float> const& baseLayerParameter, SimulationData const& data, float2 const& worldPos);
    __device__ __inline__ static float calcParameter(BaseLayerParameter<ColorVector<float>> const& baseLayerParameter, SimulationData const& data, float2 const& worldPos, int color);
    __device__ __inline__ static float calcParameter(BaseLayerParameter<ColorMatrix<float>> const& baseLayerParameter, SimulationData const& data, float2 const& worldPos, int color1, int color2);

    template <typename T, typename Parameter>
    __device__ __inline__ static T calcResultingValue(
        BaseMap const& map,
        float2 const& worldPos,
        T const& baseValue,
        T (&layerValues)[MAX_LAYERS],
        Parameter const& baseLayerParameter);

    //return -1 for base
    template <typename T>
    __device__ __inline__ static int
    getFirstMatchingLayerOrBase(SimulationData const& data, float2 const& worldPos, BaseLayerParameter<T> const& parameter);

    __device__ __inline__ static bool isCoveredByLayers(SimulationData const& data, float2 const& worldPos, LayerParameter<bool> const& enabledParameter);

    template <typename T>
    __device__ __inline__ static T calcResultingFlowField(BaseMap const& map, float2 const& worldPos, T const& baseValue, T (&layerValues)[MAX_LAYERS]);

private:

    __device__ __inline__ static float calcWeight(float2 const& delta, int const& index);
    __device__ __inline__ static float calcWeightForCircularLayer(float2 const& delta, int const& index);
    __device__ __inline__ static float calcWeightForRectLayer(float2 const& delta, int const& index);

    template<typename T>
    __device__ __inline__ static T mix(T const& baseValue, T (&layerValues)[MAX_LAYERS], float (&layerWeights)[MAX_LAYERS], int numValues);

    template <typename T>
    __device__ __inline__ static T mix(T const& baseValue, T (&layerValues)[MAX_LAYERS], float (&layerWeights)[MAX_LAYERS]);
};


/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__device__ __inline__ float ParameterCalculator::calcParameter(BaseLayerParameter<float> const& baseLayerParameter, SimulationData const& data, float2 const& worldPos)
{
    float layerValues[MAX_LAYERS];
    int numValues = 0;
    for (int i = 0; i < cudaSimulationParameters.numLayers; ++i) {
        if (baseLayerParameter.layerValues[i].enabled) {
            layerValues[numValues++] = baseLayerParameter.layerValues[i].value;
        }
    }

    return calcResultingValue(data.cellMap, worldPos, baseLayerParameter.baseValue, layerValues, baseLayerParameter);
}

__device__ __inline__ float
ParameterCalculator::calcParameter(BaseLayerParameter<ColorVector<float>> const& baseLayerParameter, SimulationData const& data, float2 const& worldPos, int color)
{
    float layerValues[MAX_LAYERS];
    int numValues = 0;
    for (int i = 0; i < cudaSimulationParameters.numLayers; ++i) {
        if (baseLayerParameter.layerValues[i].enabled) {
            layerValues[numValues++] = baseLayerParameter.layerValues[i].value[color];
        }
    }

    return calcResultingValue(data.cellMap, worldPos, baseLayerParameter.baseValue[color], layerValues, baseLayerParameter);
}

__device__ __inline__ float ParameterCalculator::calcParameter(
    BaseLayerParameter<ColorMatrix<float>> const& baseLayerParameter,
    SimulationData const& data,
    float2 const& worldPos,
    int color1,
    int color2)
{
    float layerValues[MAX_LAYERS];
    int numValues = 0;
    for (int i = 0; i < cudaSimulationParameters.numLayers; ++i) {
        if (baseLayerParameter.layerValues[i].enabled) {
            layerValues[numValues++] = baseLayerParameter.layerValues[i].value[color1][color2];
        }
    }

    return calcResultingValue(data.cellMap, worldPos, baseLayerParameter.baseValue[color1][color2], layerValues, baseLayerParameter);
}

template<typename T>
__device__ __inline__ int
ParameterCalculator::getFirstMatchingLayerOrBase(SimulationData const& data, float2 const& worldPos, BaseLayerParameter<T> const& parameter)
{
    auto const& map = data.cellMap;
    for (int i = 0; i < cudaSimulationParameters.numLayers; ++i) {
        if (parameter.layerValues[i].enabled) {
            float2 layerPos = {cudaSimulationParameters.layerPosition.layerValues[i].x, cudaSimulationParameters.layerPosition.layerValues[i].y};
            auto delta = map.getCorrectedDirection(layerPos - worldPos);
            if (calcWeight(delta, i) < NEAR_ZERO) {
                return i;
            }
        }
    }
    return -1;
}

__device__ __inline__ bool ParameterCalculator::isCoveredByLayers(SimulationData const& data, float2 const& worldPos, LayerParameter<bool> const& enabledParameter)
{
    auto const& map = data.cellMap;
    for (int i = 0; i < cudaSimulationParameters.numLayers; ++i) {
        if (enabledParameter.layerValues[i]) {
            float2 layerPos = {cudaSimulationParameters.layerPosition.layerValues[i].x, cudaSimulationParameters.layerPosition.layerValues[i].y};
            auto delta = map.getCorrectedDirection(layerPos - worldPos);
            if (calcWeight(delta, i) < NEAR_ZERO) {
                return true;
            }
        }
    }
    return false;
}

template <typename T, typename Parameter>
__device__ __inline__ T ParameterCalculator::calcResultingValue(
    BaseMap const& map,
    float2 const& worldPos,
    T const& baseValue,
    T (&layerValues)[MAX_LAYERS],
    Parameter const& baseLayerParameter)
{
    if (0 == cudaSimulationParameters.numLayers) {
        return baseValue;
    } else {
        float layerWeights[MAX_LAYERS];
        int numValues = 0;
        for (int i = 0; i < cudaSimulationParameters.numLayers; ++i) {
            if (baseLayerParameter.layerValues[i].enabled) {
                float2 layerPos = {cudaSimulationParameters.layerPosition.layerValues[i].x, cudaSimulationParameters.layerPosition.layerValues[i].y};
                auto delta = map.getCorrectedDirection(layerPos - worldPos);
                layerWeights[numValues++] = calcWeight(delta, i);
            }
        }
        return mix(baseValue, layerValues, layerWeights, numValues);
    }
}

template <typename T>
__device__ __inline__ T ParameterCalculator::calcResultingFlowField(BaseMap const& map, float2 const& worldPos, T const& baseValue, T (&layerValues)[MAX_LAYERS])
{
    if (0 == cudaSimulationParameters.numLayers) {
        return baseValue;
    } else {
        float layerWeights[MAX_LAYERS];
        int numValues = 0;
        for (int i = 0; i < cudaSimulationParameters.numLayers; ++i) {
            if (cudaSimulationParameters.layerForceFieldType.layerValues[i] != ForceField_None) {
                float2 layerPos = {cudaSimulationParameters.layerPosition.layerValues[i].x, cudaSimulationParameters.layerPosition.layerValues[i].y};
                auto delta = map.getCorrectedDirection(layerPos - worldPos);
                layerWeights[numValues++] = calcWeight(delta, i);
            }
        }
        return mix(baseValue, layerValues, layerWeights, numValues);
    }
}

__device__ __inline__ float ParameterCalculator::calcWeight(float2 const& delta, int const& index)
{
    if (cudaSimulationParameters.layerShape.layerValues[index] == LayerShapeType_Rectangular) {
        return calcWeightForRectLayer(delta, index);
    } else {
        return calcWeightForCircularLayer(delta, index);
    }
}

__device__ __inline__ float ParameterCalculator::calcWeightForCircularLayer(float2 const& delta, int const& index)
{
    auto distance = Math::length(delta);
    auto coreRadius = cudaSimulationParameters.layerCoreRadius.layerValues[index];
    auto fadeoutRadius = cudaSimulationParameters.layerFadeoutRadius.layerValues[index] + 1;

    return distance < coreRadius ? 0.0f : min(1.0f, (distance - coreRadius) / fadeoutRadius);
}

__device__ __inline__ float ParameterCalculator::calcWeightForRectLayer(float2 const& delta, int const& index)
{
    float result = 0;
    if (abs(delta.x) > cudaSimulationParameters.layerCoreRect.layerValues[index].x / 2
        || abs(delta.y) > cudaSimulationParameters.layerCoreRect.layerValues[index].y / 2) {
        float2 distanceFromRect = {
            max(0.0f, abs(delta.x) - cudaSimulationParameters.layerCoreRect.layerValues[index].x / 2),
            max(0.0f, abs(delta.y) - cudaSimulationParameters.layerCoreRect.layerValues[index].y / 2)};
        result = min(1.0f, Math::length(distanceFromRect) / (cudaSimulationParameters.layerFadeoutRadius.layerValues[index] + 1));
    }
    return result;
}

template <typename T>
__device__ __inline__ T ParameterCalculator::mix(T const& baseValue, T (&layerValues)[MAX_LAYERS], float (&layerWeights)[MAX_LAYERS], int numValues)
{
    float baseFactor = 1;
    float sum = 0;
    for (int i = 0; i < numValues; ++i) {
        baseFactor *= layerWeights[i];
        sum += 1.0f - layerWeights[i];
    }
    sum += baseFactor;
    T result = baseValue * baseFactor;
    for (int i = 0; i < numValues; ++i) {
        result += layerValues[i] * (1.0f - layerWeights[i]) / sum;
    }
    return result;
}

template <typename T>
__device__ __inline__ T ParameterCalculator::mix(T const& baseValue, T (&layerValues)[MAX_LAYERS], float (&layerWeights)[MAX_LAYERS])
{
    float baseFactor = 1;
    float sum = 0;
    for (int i = 0; i < cudaSimulationParameters.numLayers; ++i) {
        baseFactor *= layerWeights[i];
        sum += 1.0f - layerWeights[i];
    }
    sum += baseFactor;
    T result = baseValue * baseFactor;
    for (int i = 0; i < cudaSimulationParameters.numLayers; ++i) {
        result += layerValues[i] * (1.0f - layerWeights[i]) / sum;
    }
    return result;
}
