#pragma once

#include "cuda_runtime_api.h"

#include "EngineInterface/SimulationParametersSpotValues.h"
#include "ConstantMemory.cuh"

class SpotCalculator
{
public:
    __device__ static float
    calc(float SimulationParametersSpotValues::*value, SimulationData const& data, float2 const& pos)
    {
        if (0 == cudaSimulationParametersSpots.numSpots) {
            return cudaSimulationParameters.spotValues.*value;
        }
        if (1 == cudaSimulationParametersSpots.numSpots) {
            auto distance = data.cellMap.mapDistance(
                pos, {cudaSimulationParametersSpots.spots[0].posX, cudaSimulationParametersSpots.spots[0].posY});
            auto coreRadius = cudaSimulationParametersSpots.spots[0].coreRadius;
            auto fadeoutRadius = cudaSimulationParametersSpots.spots[0].fadeoutRadius + 1;
            auto factor = distance < coreRadius ? 0.0f : min(1.0f, (distance - coreRadius) / fadeoutRadius);
            return mix(
                cudaSimulationParameters.spotValues.*value,
                cudaSimulationParametersSpots.spots[0].values.*value,
                factor);
        }
        if (2 == cudaSimulationParametersSpots.numSpots) {
            auto distance1 = data.cellMap.mapDistance(
                pos, {cudaSimulationParametersSpots.spots[0].posX, cudaSimulationParametersSpots.spots[0].posY});
            auto distance2 = data.cellMap.mapDistance(
                pos, {cudaSimulationParametersSpots.spots[1].posX, cudaSimulationParametersSpots.spots[1].posY});

            auto coreRadius1 = cudaSimulationParametersSpots.spots[0].coreRadius;
            auto fadeoutRadius1 = cudaSimulationParametersSpots.spots[0].fadeoutRadius + 1;
            auto factor1 = distance1 < coreRadius1 ? 0.0f : min(1.0f, (distance1 - coreRadius1) / fadeoutRadius1);
            auto coreRadius2 = cudaSimulationParametersSpots.spots[1].coreRadius;
            auto fadeoutRadius2 = cudaSimulationParametersSpots.spots[1].fadeoutRadius + 1;
            auto factor2 = distance2 < coreRadius2 ? 0.0f : min(1.0f, (distance2 - coreRadius2) / fadeoutRadius2);

            return mix(
                cudaSimulationParameters.spotValues.*value,
                cudaSimulationParametersSpots.spots[0].values.*value,
                cudaSimulationParametersSpots.spots[1].values.*value,
                factor1,
                factor2);
        }
        return 0;
    }

private:
    __device__ static float mix(float const& a, float const& b, float factor) { return a * factor + b * (1 - factor); }

    __device__ static float mix(float const& a, float const& b, float const& c, float factor1, float factor2)
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
};