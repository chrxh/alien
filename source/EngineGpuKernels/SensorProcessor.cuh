#pragma once

#include "Cell.cuh"
#include "SimulationData.cuh"
#include "Token.cuh"

class SensorProcessor
{
public:
    __device__ __inline__ static void schedule(Token* token, SimulationData& data);

    __device__ __inline__ static void processVicinitySearches(SimulationData& data);

private:
    __device__ __inline__ static void searchVicinity(Token* token, SimulationData& data);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/
__device__ __inline__ void SensorProcessor::schedule(Token* token, SimulationData& data)
{
    auto const command = calcMod(token->memory[Enums::Sensor_Input], Enums::SensorIn_Count);

    if (command == Enums::SensorIn_SearchVicinity) {
        data.sensorOperations.tryAddEntry({token});
    }
}

__device__ __inline__ void SensorProcessor::processVicinitySearches(SimulationData& data)
{
    auto numSensorOperations = data.sensorOperations.getNumEntries();

    auto const partition = calcPartition(numSensorOperations, blockIdx.x, gridDim.x);
    for (int i = partition.startIndex; i <= partition.endIndex; ++i) {
        searchVicinity(data.sensorOperations.at(i).token, data);
    }
}

__device__ __inline__ void SensorProcessor::searchVicinity(Token* token, SimulationData& data)
{
    int minDensity = 1 + token->memory[Enums::Sensor_InMinDensity];
    int maxDensity = token->memory[Enums::Sensor_InMaxDensity];
    if (maxDensity == 0) {
        maxDensity = 255;
    }

    auto originDelta = token->sourceCell->absPos - token->cell->absPos;
    data.cellMap.mapDisplacementCorrection(originDelta);
    auto originAngle = Math::angleOfVector(originDelta);

    __shared__ uint32_t result;
    if (threadIdx.x == 0) {
        result = 0xffffffff;
    }
    __syncthreads();

    auto const partition = calcPartition(32, threadIdx.x, blockDim.x);
    for (float radius = 12.0f; radius <= cudaSimulationParameters.cellFunctionSensorRange; radius += 8.0f) {
        for (int angleIndex = partition.startIndex; angleIndex <= partition.endIndex; ++angleIndex) {
            float angle = 360.0f / 32 * angleIndex;

            auto delta = Math::unitVectorOfAngle(angle) * radius;
            auto scanPos = token->cell->absPos + delta;
            data.cellMap.mapPosCorrection(scanPos);
            auto density = static_cast<unsigned char>(data.cellFunctionData.densityMap.getDensity(scanPos));
            if (density >= minDensity && density <= maxDensity) {
                auto relAngle = Math::subtractAngle(angle, originAngle);
                uint32_t angle = static_cast<uint32_t>(QuantityConverter::convertAngleToData(relAngle + 180.0f));
                uint32_t combined = density << 8 | angle;
                atomicExch(&result, combined);
            }
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            if (result != 0xffffffff) {
                token->memory[Enums::Sensor_Output] = Enums::SensorOut_ClusterFound;
                token->memory[Enums::Sensor_OutMass] = static_cast<unsigned char>((result >> 8) & 0xff);
                auto radiusInt = static_cast<uint32_t>(radius);
                if (radiusInt > 255) {
                    radiusInt = 255;
                }
                token->memory[Enums::Sensor_OutDistance] = static_cast<unsigned char>(radiusInt);
                token->memory[Enums::Sensor_InOutAngle] = static_cast<unsigned char>(result & 0xff);
            }
        }
        if (result != 0xffffffff) {
            break;
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        if (result == 0xffffffff) {
            token->memory[Enums::Sensor_Output] = Enums::SensorOut_NothingFound;
        }
    }
}

