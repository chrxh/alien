#pragma once

#include "Cell.cuh"
#include "SimulationData.cuh"
#include "Token.cuh"

class SensorProcessor
{
public:
    __device__ __inline__ static void scheduleOperation(Token* token, SimulationData& data);

    __device__ __inline__ static void processScheduledOperation(SimulationData& data);

private:
    __device__ __inline__ static void searchVicinity(Token* token, SimulationData& data);
    __device__ __inline__ static void searchByAngle(Token* token, SimulationData& data);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/
__device__ __inline__ void SensorProcessor::scheduleOperation(Token* token, SimulationData& data)
{
    auto const command = calcMod(token->memory[Enums::Sensor_Input], Enums::SensorIn_Count);

    if (command != Enums::SensorIn_DoNothing) {
        data.sensorOperations.tryAddEntry({token});
    }
}

__device__ __inline__ void SensorProcessor::processScheduledOperation(SimulationData& data)
{
    auto numSensorOperations = data.sensorOperations.getNumEntries();

    auto const partition = calcPartition(numSensorOperations, blockIdx.x, gridDim.x);
    for (int i = partition.startIndex; i <= partition.endIndex; ++i) {
        auto token = data.sensorOperations.at(i).token;
        auto command = calcMod(token->memory[Enums::Sensor_Input], Enums::SensorIn_Count);
        if (command == Enums::SensorIn_SearchVicinity) {
            searchVicinity(token, data);
        }
        if (command == Enums::SensorIn_SearchByAngle) {
            searchByAngle(token, data);
        }
    }
}

__device__ __inline__ void SensorProcessor::searchVicinity(Token* token, SimulationData& data)
{
    __shared__ int minDensity;
    __shared__ int maxDensity;
    __shared__ int color;
    __shared__ float originAngle;
    __shared__ uint32_t result;

    if (threadIdx.x == 0) {
        minDensity = 1 + token->memory[Enums::Sensor_InMinDensity];
        maxDensity = token->memory[Enums::Sensor_InMaxDensity];
        if (maxDensity == 0) {
            maxDensity = 255;
        }
        color = calcMod(token->memory[Enums::Sensor_InColor], 7);

        auto originDelta = token->cell->absPos - token->sourceCell->absPos;
        data.cellMap.correctDirection(originDelta);
        originAngle = Math::angleOfVector(originDelta);

        result = 0xffffffff;
    }
    __syncthreads();

    auto const partition = calcPartition(32, threadIdx.x, blockDim.x);
    for (float radius = 14.0f; radius <= cudaSimulationParameters.cellFunctionSensorRange; radius += 8.0f) {
        for (int angleIndex = partition.startIndex; angleIndex <= partition.endIndex; ++angleIndex) {
            float angle = 360.0f / 32 * angleIndex;

            auto delta = Math::unitVectorOfAngle(angle) * radius;
            auto scanPos = token->cell->absPos + delta;
            data.cellMap.correctPosition(scanPos);
            auto density = static_cast<unsigned char>(data.cellFunctionData.densityMap.getDensity(scanPos, color));
            if (density >= minDensity && density <= maxDensity) {
                auto relAngle = Math::subtractAngle(angle, originAngle);
                uint32_t angle = static_cast<uint32_t>(QuantityConverter::convertAngleToData(relAngle));
                uint32_t combined = density << 8 | angle;
                atomicExch(&result, combined);
            }
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            if (result != 0xffffffff) {
                token->memory[Enums::Sensor_Output] = Enums::SensorOut_ClusterFound;
                token->memory[Enums::Sensor_OutDensity] = static_cast<unsigned char>((result >> 8) & 0xff);
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

__device__ __inline__ void SensorProcessor::searchByAngle(Token* token, SimulationData& data)
{
    __shared__ int minDensity;
    __shared__ int maxDensity;
    __shared__ int color;
    __shared__ float2 searchDelta;
    __shared__ uint32_t result;

    if (threadIdx.x == 0) {
        minDensity = 1 + token->memory[Enums::Sensor_InMinDensity];
        maxDensity = token->memory[Enums::Sensor_InMaxDensity];
        if (maxDensity == 0) {
            maxDensity = 255;
        }
        color = calcMod(token->memory[Enums::Sensor_InColor], 7);

        searchDelta = token->cell->absPos - token->sourceCell->absPos;
        data.cellMap.correctDirection(searchDelta);
        auto angle = QuantityConverter::convertDataToAngle(token->memory[Enums::Sensor_InOutAngle]);
        Math::normalize(searchDelta);
        searchDelta = Math::rotateClockwise(searchDelta, angle);

        result = 0xffffffff;
    }
    __syncthreads();

    auto const partition = calcPartition(64, threadIdx.x, blockDim.x);
    for (int distanceIndex = partition.startIndex; distanceIndex <= partition.endIndex; ++distanceIndex) {
        auto distance = 12.0f + cudaSimulationParameters.cellFunctionSensorRange / 64 * distanceIndex;
        auto scanPos = token->cell->absPos + searchDelta * distance;
        data.cellMap.correctPosition(scanPos);
        auto density = static_cast<unsigned char>(data.cellFunctionData.densityMap.getDensity(scanPos, color));
        if (density >= minDensity && density <= maxDensity) {
            uint32_t combined = static_cast<uint32_t>(distance) << 8 | static_cast<uint32_t>(density);
            atomicMin(&result, combined);
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        if (result != 0xffffffff) {
            token->memory[Enums::Sensor_Output] = Enums::SensorOut_ClusterFound;
            token->memory[Enums::Sensor_OutDensity] = static_cast<unsigned char>(result & 0xff);
            auto distance = static_cast<uint32_t>((result >> 8) & 0xff);
            if (distance > 255) {
                distance = 255;
            }
            token->memory[Enums::Sensor_OutDistance] = static_cast<unsigned char>(distance);
        } else {
            token->memory[Enums::Sensor_Output] = Enums::SensorOut_NothingFound;
        }
    }
}

