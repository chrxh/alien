#pragma once

#include "Cell.cuh"
#include "SimulationData.cuh"
#include "CellFunctionProcessor.cuh"

class SensorProcessor
{
public:
    __inline__ __device__ static void process(SimulationData& data, SimulationStatistics& statistics);

private:
    static int constexpr NumScanAngles = 32;
    static int constexpr NumScanPoints = 64;

    __inline__ __device__ static void processCell(SimulationData& data, SimulationStatistics& statistics, Cell* cell);
    __inline__ __device__ static void searchNeighborhood(SimulationData& data, SimulationStatistics& statistics, Cell* cell, Activity& activity);
    __inline__ __device__ static void
    searchByAngle(SimulationData& data, SimulationStatistics& statistics, Cell* cell, Activity& activity);

    __inline__ __device__ static uint8_t convertAngleToData(float angle);
    __inline__ __device__ static float convertDataToAngle(uint8_t b);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__inline__ __device__ void SensorProcessor::process(SimulationData& data, SimulationStatistics& statistics)
{
    auto& operations = data.cellFunctionOperations[CellFunction_Sensor];
    auto partition = calcBlockPartition(operations.getNumEntries());
    for (int i = partition.startIndex; i <= partition.endIndex; ++i) {
        processCell(data, statistics, operations.at(i).cell);
    }
}

__inline__ __device__ void SensorProcessor::processCell(SimulationData& data, SimulationStatistics& statistics, Cell* cell)
{
    __shared__ Activity activity;
    if (threadIdx.x == 0) {
        activity = CellFunctionProcessor::calcInputActivity(cell);
    }
    __syncthreads();

    if (abs(activity.channels[0]) > cudaSimulationParameters.cellFunctionSensorActivityThreshold) {
        statistics.incNumSensorActivities(cell->color);
        switch (cell->cellFunctionData.sensor.mode) {
        case SensorMode_Neighborhood: {
            searchNeighborhood(data, statistics, cell, activity);
        } break;
        case SensorMode_FixedAngle: {
            searchByAngle(data, statistics, cell, activity);
        } break;
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        CellFunctionProcessor::setActivity(cell, activity);
    }
}

__inline__ __device__ void
SensorProcessor::searchNeighborhood(SimulationData& data, SimulationStatistics& statistics, Cell* cell, Activity& activity)
{
    __shared__ int minDensity;
    __shared__ int color;
    __shared__ float refScanAngle;
    __shared__ uint64_t lookupResult;

    if (threadIdx.x == 0) {
        refScanAngle = Math::angleOfVector(CellFunctionProcessor::calcSignalDirection(data, cell));
        minDensity = toInt(cell->cellFunctionData.sensor.minDensity * 100);
        color = cell->cellFunctionData.sensor.color;

        lookupResult = 0xffffffffffffffff;
    }
    __syncthreads();

    auto const partition = calcPartition(NumScanAngles, threadIdx.x, blockDim.x);
    auto startRadius = color == cell->color ? 14.0f : 0.0f;
    for (float radius = startRadius; radius <= cudaSimulationParameters.cellFunctionSensorRange[cell->color]; radius += 8.0f) {
        for (int angleIndex = partition.startIndex; angleIndex <= partition.endIndex; ++angleIndex) {
            float angle = 360.0f / NumScanAngles * angleIndex;

            auto delta = Math::unitVectorOfAngle(angle) * radius;
            auto scanPos = cell->pos + delta;
            data.cellMap.correctPosition(scanPos);
            auto density = static_cast<unsigned char>(data.preprocessedCellFunctionData.densityMap.getDensity(scanPos, color));
            if (density >= minDensity) {
                uint32_t creatureId = [&] {
                    if (cudaSimulationParameters.cellFunctionAttackerSensorDetectionFactor[cell->color] > NEAR_ZERO) {
                        for (float dx = -3.0f; dx < 3.0f + NEAR_ZERO; dx += 1.0f) {
                            for (float dy = -3.0f; dy < 3.0f + NEAR_ZERO; dy += 1.0f) {
                                auto otherCell = data.cellMap.getFirst(scanPos + float2{dx, dy});
                                if (otherCell && otherCell->color == color) {
                                    return static_cast<uint32_t>(otherCell->creatureId);
                                }
                            }
                        }
                    }
                    return 0xffffffff;
                }();
                auto relAngle = Math::subtractAngle(angle, refScanAngle);
                uint32_t angle = convertAngleToData(relAngle);
                uint64_t combined =
                    static_cast<uint64_t>(radius) << 48 | static_cast<uint64_t>(density) << 40 | static_cast<uint64_t>(angle) << 32 | creatureId;
                alienAtomicMin64(&lookupResult, combined);
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        if (lookupResult != 0xffffffffffffffff) {
            activity.channels[0] = 1;                                                     //something found
            activity.channels[1] = static_cast<float>((lookupResult >> 40) & 0xff) / 256;  //density
            activity.channels[2] = static_cast<float>(lookupResult >> 48) / 256;  //distance
            activity.channels[3] = convertDataToAngle(static_cast<int8_t>((lookupResult >> 32) & 0xff)) / 360.0f;  //angle: between -0.5 and 0.5
            auto targetCreatureId = lookupResult & 0xffffffff;
            if (targetCreatureId != 0xffffffff) {
                cell->cellFunctionData.sensor.targetedCreatureId = toInt(targetCreatureId);
            }
            statistics.incNumSensorMatches(cell->color);
        } else {
            activity.channels[0] = 0;  //nothing found
        }
    }
    __syncthreads();
}

__inline__ __device__ void
SensorProcessor::searchByAngle(SimulationData& data, SimulationStatistics& statistics, Cell* cell, Activity& activity)
{
    __shared__ int minDensity;
    __shared__ int color;
    __shared__ float2 searchDelta;
    __shared__ uint64_t lookupResult;

    if (threadIdx.x == 0) {
        minDensity = toInt(cell->cellFunctionData.sensor.minDensity * 255);
        color = cell->cellFunctionData.sensor.color;

        searchDelta = CellFunctionProcessor::calcSignalDirection(data, cell);
        searchDelta = Math::rotateClockwise(searchDelta, cell->cellFunctionData.sensor.angle);

        lookupResult = 0xffffffffffffffff;
    }
    __syncthreads();

    auto const partition = calcPartition(NumScanPoints, threadIdx.x, blockDim.x);
    for (int distanceIndex = partition.startIndex; distanceIndex <= partition.endIndex; ++distanceIndex) {
        auto distance = 14.0f + cudaSimulationParameters.cellFunctionSensorRange[cell->color] / NumScanPoints * distanceIndex;
        auto scanPos = cell->pos + searchDelta * distance;
        data.cellMap.correctPosition(scanPos);
        auto density = static_cast<unsigned char>(data.preprocessedCellFunctionData.densityMap.getDensity(scanPos, color));
        if (density >= minDensity) {
            uint32_t creatureId = [&] {
                if (cudaSimulationParameters.cellFunctionAttackerSensorDetectionFactor[cell->color] > NEAR_ZERO) {
                    for (float dx = -3.0f; dx < 3.0f + NEAR_ZERO; dx += 1.0f) {
                        for (float dy = -3.0f; dy < 3.0f + NEAR_ZERO; dy += 1.0f) {
                            auto otherCell = data.cellMap.getFirst(scanPos + float2{dx, dy});
                            if (otherCell && otherCell->color == color) {
                                return static_cast<uint32_t>(otherCell->creatureId);
                            }
                        }
                    }
                }
                return 0xffffffff;
            }();

            uint64_t combined = static_cast<uint64_t>(distance) << 48 | static_cast<uint64_t>(density) << 40 | creatureId;
            alienAtomicMin64(&lookupResult, combined);

        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        if (lookupResult != 0xffffffffffffffff) {
            activity.channels[0] = 1;                                                     //something found
            activity.channels[1] = static_cast<float>((lookupResult >> 40) & 0xff) / 256;  //density
            activity.channels[2] = static_cast<float>(lookupResult >> 48) / 256;           //distance
            auto targetCreatureId = lookupResult & 0xffffffff;
            if (targetCreatureId != 0xffffffff) {
                cell->cellFunctionData.sensor.targetedCreatureId = toInt(targetCreatureId);
            }
            statistics.incNumSensorMatches(cell->color);
        } else {
            activity.channels[0] = 0;  //nothing found
        }
    }
}

__inline__ __device__ uint8_t SensorProcessor::convertAngleToData(float angle)
{
    //0 to 180 degree => 0 to 128
    //-180 to 0 degree => 128 to 256 (= 0)
    angle = remainderf(remainderf(angle, 360.0f) + 360.0f, 360.0f);  //get angle between 0 and 360
    if (angle > 180.0f) {
        angle -= 360.0f;
    }
    int result = static_cast<int>(angle * 128.0f / 180.0f);
    return static_cast<uint8_t>(result);
}

__inline__ __device__ float SensorProcessor::convertDataToAngle(uint8_t b)
{
    //0 to 127 => 0 to 179 degree
    //128 to 255 => -179 to 0 degree
    if (b < 128) {
        return (0.5f + static_cast<float>(b)) * (180.0f / 128.0f);
    } else {
        return (-256.0f - 0.5f + static_cast<float>(b)) * (180.0f / 128.0f);
    }
}
