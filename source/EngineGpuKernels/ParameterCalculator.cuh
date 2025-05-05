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
    __device__ __inline__ static float calcParameter(BaseLayerParameter<float> const& parameter, SimulationData const& data, float2 const& worldPos);
    __device__ __inline__ static float calcParameter(BaseLayerParameter<ColorVector<float>> const& parameter, SimulationData const& data, float2 const& worldPos, int color);
    __device__ __inline__ static float calcParameter(BaseLayerParameter<ColorMatrix<float>> const& parameter, SimulationData const& data, float2 const& worldPos, int color1, int color2);
    __device__ __inline__ static float2 calcParameter(float2 const& baseValue, float2 (&layerValues)[MAX_LAYERS], SimulationData const& data, float2 const& worldPos);
    __device__ __inline__ static FloatColorRGB calcParameter(BaseLayerParameter<FloatColorRGB> const& parameter, BaseMap const& map, float2 const& worldPos);

    //return -1 for base
    template <typename T>
    __device__ __inline__ static int
    getFirstMatchingLayerOrBase(SimulationData const& data, float2 const& worldPos, BaseLayerParameter<T> const& parameter);

    __device__ __inline__ static bool isCoveredByLayers(SimulationData const& data, float2 const& worldPos, LayerParameter<bool> const& enabledParameter);


private:

    __device__ __inline__ static float calcWeight(float2 const& delta, int const& index);
    __device__ __inline__ static float calcWeightForCircularLayer(float2 const& delta, int const& index);
    __device__ __inline__ static float calcWeightForRectLayer(float2 const& delta, int const& index);
};


/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__device__ __inline__ float ParameterCalculator::calcParameter(BaseLayerParameter<float> const& parameter, SimulationData const& data, float2 const& worldPos)
{
    auto result = parameter.baseValue;
    for (int i = 0; i < cudaSimulationParameters.numLayers; ++i) {
        if (parameter.layerValues[i].enabled) {
            float2 layerPos = {cudaSimulationParameters.layerPosition.layerValues[i].x, cudaSimulationParameters.layerPosition.layerValues[i].y};
            auto delta = data.cellMap.getCorrectedDirection(layerPos - worldPos);
            auto weight = calcWeight(delta, i);
            result = result * weight + parameter.layerValues[i].value * (1.0f - weight);
        }
    }
    return result;
}

__device__ __inline__ float ParameterCalculator::calcParameter(BaseLayerParameter<ColorVector<float>> const& parameter, SimulationData const& data, float2 const& worldPos, int color)
{
    auto result = parameter.baseValue[color];
    for (int i = 0; i < cudaSimulationParameters.numLayers; ++i) {
        if (parameter.layerValues[i].enabled) {
            float2 layerPos = {cudaSimulationParameters.layerPosition.layerValues[i].x, cudaSimulationParameters.layerPosition.layerValues[i].y};
            auto delta = data.cellMap.getCorrectedDirection(layerPos - worldPos);
            auto weight = calcWeight(delta, i);
            result = result * weight + parameter.layerValues[i].value[color] * (1.0f - weight);
        }
    }
    return result;
}

__device__ __inline__ float ParameterCalculator::calcParameter(
    BaseLayerParameter<ColorMatrix<float>> const& parameter,
    SimulationData const& data,
    float2 const& worldPos,
    int color1,
    int color2)
{
    auto result = parameter.baseValue[color1][color2];
    for (int i = 0; i < cudaSimulationParameters.numLayers; ++i) {
        if (parameter.layerValues[i].enabled) {
            float2 layerPos = {cudaSimulationParameters.layerPosition.layerValues[i].x, cudaSimulationParameters.layerPosition.layerValues[i].y};
            auto delta = data.cellMap.getCorrectedDirection(layerPos - worldPos);
            auto weight = calcWeight(delta, i);
            result = result * weight + parameter.layerValues[i].value[color1][color2] * (1.0f - weight);
        }
    }
    return result;
}

__device__ __inline__ float2 ParameterCalculator::calcParameter(float2 const& baseValue, float2(& layerValues)[MAX_LAYERS], SimulationData const& data, float2 const& worldPos)
{
    auto result = baseValue;
    for (int i = 0; i < cudaSimulationParameters.numLayers; ++i) {
        float2 layerPos = {cudaSimulationParameters.layerPosition.layerValues[i].x, cudaSimulationParameters.layerPosition.layerValues[i].y};
        auto delta = data.cellMap.getCorrectedDirection(layerPos - worldPos);
        auto weight = calcWeight(delta, i);
        result = result * weight + layerValues[i] * (1.0f - weight);
    }
    return result;
}

__device__ __inline__ FloatColorRGB ParameterCalculator::calcParameter(BaseLayerParameter<FloatColorRGB> const& parameter, BaseMap const& map, float2 const& worldPos)
{
    auto result = parameter.baseValue;
    for (int i = 0; i < cudaSimulationParameters.numLayers; ++i) {
        if (parameter.layerValues[i].enabled) {
            float2 layerPos = {cudaSimulationParameters.layerPosition.layerValues[i].x, cudaSimulationParameters.layerPosition.layerValues[i].y};
            auto delta = map.getCorrectedDirection(layerPos - worldPos);
            auto weight = calcWeight(delta, i);
            result.r = result.r * weight + parameter.layerValues[i].value.r * (1.0f - weight);
            result.g = result.g * weight + parameter.layerValues[i].value.g * (1.0f - weight);
            result.b = result.b * weight + parameter.layerValues[i].value.b * (1.0f - weight);
        }
    }
    return result;
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
