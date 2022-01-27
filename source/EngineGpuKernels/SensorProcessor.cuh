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
    __device__ __inline__ static unsigned char calcDensity(float2 const& pos, SimulationData& data);
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
    int minDensity = token->memory[Enums::Sensor_InMinMass];
    int maxDensity = token->memory[Enums::Sensor_InMaxMass];

    auto originDelta = token->sourceCell->absPos - token->cell->absPos;
    data.cellMap.mapDisplacementCorrection(originDelta);
    auto originAngle = Math::angleOfVector(originDelta);

    __shared__ uint32_t result;
    if (threadIdx.x == 0) {
        result = 0xffffffff;
    }
    __syncthreads();

    auto const partition = calcPartition(64 * 64, threadIdx.x, blockDim.x);
    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        int indexX = index % 64;
        int indexY = index / 64;
        float dx = -256.0f + toFloat(indexX) * 8;
        float dy = -256.0f + toFloat(indexY) * 8;
        auto radiusSquared = dx * dx + dy * dy;
        if (radiusSquared < 256.0f * 256.0f && radiusSquared >= 3.0f * 3.0f) {

            auto scanPos = token->cell->absPos + float2{dx, dy};
            auto scannedCell = data.cellMap.getFirst(scanPos);
            if (scannedCell) {
                auto density = calcDensity(scanPos, data);
                if (density >= minDensity && density <= maxDensity) {
                    auto radius = static_cast<uint32_t>(sqrt(radiusSquared));
                    auto resultAngle = Math::angleOfVector(float2{dx, dy});
                    auto relAngle = Math::subtractAngle(resultAngle, originAngle);
                    uint32_t angle = static_cast<uint32_t>(QuantityConverter::convertAngleToData(relAngle));
                    uint32_t combined = radius << 16 | density << 8 | angle;
                    atomicMin(&result, combined);
                }
            }
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        if (result == 0xffffffff) {
            token->memory[Enums::Sensor_Output] = Enums::SensorOut_NothingFound;
        } else {
            token->memory[Enums::Sensor_Output] = Enums::SensorOut_ClusterFound;
            token->memory[Enums::Sensor_OutMass] = static_cast<unsigned char>((result >> 8) & 0xff);
            token->memory[Enums::Sensor_OutDistance] = static_cast<unsigned char>((result >> 16) & 0xff);
            token->memory[Enums::Sensor_InOutAngle] = static_cast<unsigned char>(result & 0xff);
        }
    }
}

__device__ __inline__ unsigned char SensorProcessor::calcDensity(float2 const& pos, SimulationData& data)
{
    int numCells = 0;
    for (float dx = -6.0f; dx <= 6.01f; dx += 2.0f) {
        for (float dy = -6.0f; dy <= -6.01f; dy += 1.0f) {
            if (data.cellMap.getFirst(pos + float2{dx, dy})) {
                ++numCells;
            }
        }
    }
    return static_cast<unsigned char>(numCells * 255 / 100);
}
