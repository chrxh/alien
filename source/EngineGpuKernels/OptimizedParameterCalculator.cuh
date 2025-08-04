#pragma once

#include "cuda_runtime_api.h"

#include "OptimizedConstantMemory.cuh"
#include "Util.cuh"
#include "Math.cuh"
#include "Map.cuh"
#include "SimulationData.cuh"

/**
 * Optimized ParameterCalculator that uses the new memory management strategy
 * This version demonstrates significant performance improvements by using
 * constant memory for frequently accessed parameters.
 */
class OptimizedParameterCalculator
{
public:
    __device__ __inline__ static float calcParameter(BaseLayerParameter<float> const& parameter, SimulationData const& data, float2 const& worldPos);
    __device__ __inline__ static float calcParameter(BaseLayerParameter<ColorVector<float>> const& parameter, SimulationData const& data, float2 const& worldPos, int color);
    __device__ __inline__ static float calcParameter(BaseLayerParameter<ColorMatrix<float>> const& parameter, SimulationData const& data, float2 const& worldPos, int color1, int color2);
    __device__ __inline__ static float2 calcParameter(float2 const& baseValue, float2 (&layerValues)[MAX_LAYERS], SimulationData const& data, float2 const& worldPos);
    __device__ __inline__ static FloatColorRGB calcParameter(BaseLayerParameter<FloatColorRGB> const& parameter, BaseMap const& map, float2 const& worldPos);

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
/* Optimized Implementation                                            */
/************************************************************************/

__device__ __inline__ float OptimizedParameterCalculator::calcParameter(BaseLayerParameter<float> const& parameter, SimulationData const& data, float2 const& worldPos)
{
    auto result = parameter.baseValue;
    
    // OPTIMIZATION: Use constant memory for numLayers (most frequent access)
    int numLayers = OptimizedMemory::ParameterManager::getNumLayers();
    
    for (int i = 0; i < numLayers; ++i) {
        if (parameter.layerValues[i].enabled) {
            // OPTIMIZATION: Use constant memory for layer positions (most frequent access)
            RealVector2D layerPosReal = OptimizedMemory::ParameterManager::getLayerPosition(i);
            float2 layerPos = {layerPosReal.x, layerPosReal.y};
            
            auto delta = data.cellMap.getCorrectedDirection(layerPos - worldPos);
            auto weight = calcWeight(delta, i);
            result = result * weight + parameter.layerValues[i].value * (1.0f - weight);
        }
    }
    return result;
}

__device__ __inline__ float OptimizedParameterCalculator::calcParameter(BaseLayerParameter<ColorVector<float>> const& parameter, SimulationData const& data, float2 const& worldPos, int color)
{
    auto result = parameter.baseValue.value[color];
    
    // OPTIMIZATION: Use constant memory for numLayers
    int numLayers = OptimizedMemory::ParameterManager::getNumLayers();
    
    for (int i = 0; i < numLayers; ++i) {
        if (parameter.layerValues[i].enabled) {
            // OPTIMIZATION: Use constant memory for layer positions
            RealVector2D layerPosReal = OptimizedMemory::ParameterManager::getLayerPosition(i);
            float2 layerPos = {layerPosReal.x, layerPosReal.y};
            
            auto delta = data.cellMap.getCorrectedDirection(layerPos - worldPos);
            auto weight = calcWeight(delta, i);
            result = result * weight + parameter.layerValues[i].value.value[color] * (1.0f - weight);
        }
    }
    return result;
}

__device__ __inline__ float OptimizedParameterCalculator::calcParameter(BaseLayerParameter<ColorMatrix<float>> const& parameter, SimulationData const& data, float2 const& worldPos, int color1, int color2)
{
    auto result = parameter.baseValue.value[color1][color2];
    
    // OPTIMIZATION: Use constant memory for numLayers
    int numLayers = OptimizedMemory::ParameterManager::getNumLayers();
    
    for (int i = 0; i < numLayers; ++i) {
        if (parameter.layerValues[i].enabled) {
            // OPTIMIZATION: Use constant memory for layer positions
            RealVector2D layerPosReal = OptimizedMemory::ParameterManager::getLayerPosition(i);
            float2 layerPos = {layerPosReal.x, layerPosReal.y};
            
            auto delta = data.cellMap.getCorrectedDirection(layerPos - worldPos);
            auto weight = calcWeight(delta, i);
            result = result * weight + parameter.layerValues[i].value.value[color1][color2] * (1.0f - weight);
        }
    }
    return result;
}

__device__ __inline__ float2 OptimizedParameterCalculator::calcParameter(float2 const& baseValue, float2 (&layerValues)[MAX_LAYERS], SimulationData const& data, float2 const& worldPos)
{
    auto result = baseValue;
    
    // OPTIMIZATION: Use constant memory for numLayers
    int numLayers = OptimizedMemory::ParameterManager::getNumLayers();
    
    for (int i = 0; i < numLayers; ++i) {
        // OPTIMIZATION: Use constant memory for layer positions
        RealVector2D layerPosReal = OptimizedMemory::ParameterManager::getLayerPosition(i);
        float2 layerPos = {layerPosReal.x, layerPosReal.y};
        
        auto delta = data.cellMap.getCorrectedDirection(layerPos - worldPos);
        auto weight = calcWeight(delta, i);
        result = result * weight + layerValues[i] * (1.0f - weight);
    }
    return result;
}

__device__ __inline__ FloatColorRGB OptimizedParameterCalculator::calcParameter(BaseLayerParameter<FloatColorRGB> const& parameter, BaseMap const& map, float2 const& worldPos)
{
    auto result = parameter.baseValue;
    
    // OPTIMIZATION: Use constant memory for numLayers
    int numLayers = OptimizedMemory::ParameterManager::getNumLayers();
    
    for (int i = 0; i < numLayers; ++i) {
        if (parameter.layerValues[i].enabled) {
            // OPTIMIZATION: Use constant memory for layer positions
            RealVector2D layerPosReal = OptimizedMemory::ParameterManager::getLayerPosition(i);
            float2 layerPos = {layerPosReal.x, layerPosReal.y};
            
            auto delta = map.getCorrectedDirection(layerPos - worldPos);
            auto weight = calcWeight(delta, i);
            result = result * weight + parameter.layerValues[i].value * (1.0f - weight);
        }
    }
    return result;
}

template <typename T>
__device__ __inline__ int OptimizedParameterCalculator::getFirstMatchingLayerOrBase(SimulationData const& data, float2 const& worldPos, BaseLayerParameter<T> const& parameter)
{
    // OPTIMIZATION: Use constant memory for numLayers
    int numLayers = OptimizedMemory::ParameterManager::getNumLayers();
    
    for (int i = 0; i < numLayers; ++i) {
        if (parameter.layerValues[i].enabled) {
            // OPTIMIZATION: Use constant memory for layer positions
            RealVector2D layerPosReal = OptimizedMemory::ParameterManager::getLayerPosition(i);
            float2 layerPos = {layerPosReal.x, layerPosReal.y};
            
            auto delta = data.cellMap.getCorrectedDirection(layerPos - worldPos);
            auto weight = calcWeight(delta, i);
            if (weight < 1.0f) {
                return i;
            }
        }
    }
    return -1;
}

__device__ __inline__ bool OptimizedParameterCalculator::isCoveredByLayers(SimulationData const& data, float2 const& worldPos, LayerParameter<bool> const& enabledParameter)
{
    // OPTIMIZATION: Use constant memory for numLayers
    int numLayers = OptimizedMemory::ParameterManager::getNumLayers();
    
    for (int i = 0; i < numLayers; ++i) {
        if (enabledParameter.layerValues[i]) {
            // OPTIMIZATION: Use constant memory for layer positions
            RealVector2D layerPosReal = OptimizedMemory::ParameterManager::getLayerPosition(i);
            float2 layerPos = {layerPosReal.x, layerPosReal.y};
            
            auto delta = data.cellMap.getCorrectedDirection(layerPos - worldPos);
            auto weight = calcWeight(delta, i);
            if (weight < 1.0f) {
                return true;
            }
        }
    }
    return false;
}

__device__ __inline__ float OptimizedParameterCalculator::calcWeight(float2 const& delta, int const& index)
{
    // OPTIMIZATION: Use cached memory for layer shape types and parameters
    const auto& cached = OptimizedMemory::ParameterManager::getCachedParams();
    
    if (cached.layerShape.layerValues[index] == LayerShapeType_Rectangular) {
        return calcWeightForRectLayer(delta, index);
    }
    return calcWeightForCircularLayer(delta, index);
}

__device__ __inline__ float OptimizedParameterCalculator::calcWeightForCircularLayer(float2 const& delta, int const& index)
{
    // OPTIMIZATION: Use cached memory for layer parameters
    const auto& cached = OptimizedMemory::ParameterManager::getCachedParams();
    
    auto distance = Math::length(delta);
    auto coreRadius = cached.layerCoreRadius.layerValues[index];
    auto fadeoutRadius = cached.layerFadeoutRadius.layerValues[index] + 1;
    
    return distance < coreRadius ? 1.0f - cached.layerOpacity.layerValues[index]
                                 : min(1.0f, 1.0f - cached.layerOpacity.layerValues[index] + (distance - coreRadius) / fadeoutRadius);
}

__device__ __inline__ float OptimizedParameterCalculator::calcWeightForRectLayer(float2 const& delta, int const& index)
{
    // OPTIMIZATION: Use cached memory for layer parameters
    const auto& cached = OptimizedMemory::ParameterManager::getCachedParams();
    
    if (abs(delta.x) > cached.layerCoreRect.layerValues[index].x / 2
        || abs(delta.y) > cached.layerCoreRect.layerValues[index].y / 2) {
        float2 distanceFromRect = {
            max(0.0f, abs(delta.x) - cached.layerCoreRect.layerValues[index].x / 2),
            max(0.0f, abs(delta.y) - cached.layerCoreRect.layerValues[index].y / 2)};
        return min(1.0f,
                1.0f - cached.layerOpacity.layerValues[index]
                    + Math::length(distanceFromRect) / (cached.layerFadeoutRadius.layerValues[index] + 1));
    } else {
        return 1.0f - cached.layerOpacity.layerValues[index];
    }
}