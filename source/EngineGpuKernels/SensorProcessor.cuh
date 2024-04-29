#pragma once

#include "Object.cuh"
#include "SimulationData.cuh"
#include "CellFunctionProcessor.cuh"

class SensorProcessor
{
public:
    __inline__ __device__ static void process(SimulationData& data, SimulationStatistics& statistics);

private:
    static int constexpr NumScanAngles = 32;
    static int constexpr NumScanPoints = 64;
    static int constexpr ScanStep = 8.0f;

    __inline__ __device__ static void processCell(SimulationData& data, SimulationStatistics& statistics, Cell* cell);
    __inline__ __device__ static void searchNeighborhood(SimulationData& data, SimulationStatistics& statistics, Cell* cell, Activity& activity);
    __inline__ __device__ static void searchByAngle(SimulationData& data, SimulationStatistics& statistics, Cell* cell, Activity& activity);

    __inline__ __device__ static uint32_t getCreatureId(SimulationData& data, Cell* cell, float2 const& scanPos, float& preciseDistance);

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
    __shared__ uint32_t minDensity;
    __shared__ uint8_t colorRestriction;
    __shared__ bool otherMutantsRestriction;
    __shared__ float refScanAngle;
    __shared__ uint64_t lookupResult;

    if (threadIdx.x == 0) {
        refScanAngle = Math::angleOfVector(CellFunctionProcessor::calcSignalDirection(data, cell));
        minDensity = toInt(cell->cellFunctionData.sensor.minDensity * 100);
        colorRestriction = cell->cellFunctionData.sensor.restrictToColor;
        otherMutantsRestriction = cell->cellFunctionData.sensor.restrictToOtherMutants;
        lookupResult = 0xffffffffffffffff;
    }
    __syncthreads();

    auto const partition = calcPartition(NumScanAngles, threadIdx.x, blockDim.x);
    auto startRadius = (colorRestriction == 255 || colorRestriction == cell->color) ? 14.0f : 0.0f;
    auto const& densityMap = data.preprocessedCellFunctionData.densityMap;
    for (float radius = startRadius; radius <= cudaSimulationParameters.cellFunctionSensorRange[cell->color]; radius += ScanStep) {
        for (int angleIndex = partition.startIndex; angleIndex <= partition.endIndex; ++angleIndex) {
            float angle = 360.0f / NumScanAngles * angleIndex;

            auto delta = Math::unitVectorOfAngle(angle) * radius;
            auto scanPos = cell->pos + delta;
            data.cellMap.correctPosition(scanPos);

            uint32_t density;
            if (colorRestriction != 255 && !otherMutantsRestriction) {
                density = densityMap.getColorDensity(scanPos, colorRestriction);
            }
            if (colorRestriction == 255 && otherMutantsRestriction) {
                density = densityMap.getOtherMutantsDensity(scanPos, cell->mutationId);
            }
            if (colorRestriction != 255 && otherMutantsRestriction) {
                density =
                    min(densityMap.getOtherMutantsDensity(scanPos, cell->mutationId),
                        densityMap.getColorDensity(scanPos, colorRestriction));
            }
            if (colorRestriction == 255 && !otherMutantsRestriction) {
                density = densityMap.getCellDensity(scanPos);
            }
            if (density < minDensity) {
                continue;
            }
            float preciseAngle = angle;
            float preciseDistance = radius;
            uint32_t creatureId = getCreatureId(data, cell, scanPos, preciseDistance);
            auto relAngle = Math::subtractAngle(preciseAngle, refScanAngle);
            uint32_t relAngleData = convertAngleToData(relAngle);
            uint64_t combined = 
                static_cast<uint64_t>(preciseDistance) << 48 | static_cast<uint64_t>(density) << 40 | static_cast<uint64_t>(relAngleData) << 32 | creatureId;
            alienAtomicMin64(&lookupResult, combined);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        if (lookupResult != 0xffffffffffffffff) {
            activity.channels[0] = 1;                                                     //something found
            activity.channels[1] = static_cast<float>((lookupResult >> 40) & 0xff) / 256;  //density
            activity.channels[2] = 1.0f - min(1.0f, static_cast<float>(lookupResult >> 48) / 256);  //distance: 1 = close, 0 = far away
            activity.channels[3] = convertDataToAngle(static_cast<int8_t>((lookupResult >> 32) & 0xff)) / 360.0f;  //angle: between -0.5 and 0.5
            cell->cellFunctionData.sensor.memoryChannel1 = activity.channels[1];
            cell->cellFunctionData.sensor.memoryChannel2 = activity.channels[2];
            cell->cellFunctionData.sensor.memoryChannel3 = activity.channels[3];
            auto targetCreatureId = lookupResult & 0xffffffff;
            if (targetCreatureId != 0xffffffff) {
                cell->cellFunctionData.sensor.targetedCreatureId = static_cast<uint32_t>(targetCreatureId);
            }
            statistics.incNumSensorMatches(cell->color);
        } else {
            activity.channels[0] = 0;  //nothing found
            activity.channels[1] = cell->cellFunctionData.sensor.memoryChannel1;
            activity.channels[2] = cell->cellFunctionData.sensor.memoryChannel2;
            activity.channels[3] = cell->cellFunctionData.sensor.memoryChannel3;
        }
    }
    __syncthreads();
}

__inline__ __device__ void
SensorProcessor::searchByAngle(SimulationData& data, SimulationStatistics& statistics, Cell* cell, Activity& activity)
{
    __shared__ uint32_t minDensity;
    __shared__ uint8_t colorRestriction;
    __shared__ bool otherMutantsRestriction;
    __shared__ float2 searchDelta;
    __shared__ uint64_t lookupResult;

    if (threadIdx.x == 0) {
        minDensity = toInt(cell->cellFunctionData.sensor.minDensity * 255);
        colorRestriction = cell->cellFunctionData.sensor.restrictToColor;
        otherMutantsRestriction = cell->cellFunctionData.sensor.restrictToOtherMutants;
        searchDelta = CellFunctionProcessor::calcSignalDirection(data, cell);
        searchDelta = Math::rotateClockwise(searchDelta, cell->cellFunctionData.sensor.angle);

        lookupResult = 0xffffffffffffffff;
    }
    __syncthreads();

    auto const partition = calcPartition(NumScanPoints, threadIdx.x, blockDim.x);
    auto startRadius = (colorRestriction == 255 || colorRestriction == cell->color) ? 14.0f : 0.0f;
    auto const& densityMap = data.preprocessedCellFunctionData.densityMap;
    for (int distanceIndex = partition.startIndex; distanceIndex <= partition.endIndex; ++distanceIndex) {
        auto distance = startRadius + cudaSimulationParameters.cellFunctionSensorRange[cell->color] / NumScanPoints * distanceIndex;
        auto scanPos = cell->pos + searchDelta * distance;
        data.cellMap.correctPosition(scanPos);

        uint32_t density;
        if (colorRestriction != 255 && !otherMutantsRestriction) {
            density = densityMap.getColorDensity(scanPos, colorRestriction);
        }
        if (colorRestriction == 255 && otherMutantsRestriction) {
            density = densityMap.getOtherMutantsDensity(scanPos, cell->mutationId);
        }
        if (colorRestriction != 255 && otherMutantsRestriction) {
            density = min(densityMap.getOtherMutantsDensity(scanPos, cell->mutationId), densityMap.getColorDensity(scanPos, colorRestriction));
        }
        if (colorRestriction == 255 && !otherMutantsRestriction) {
            density = densityMap.getCellDensity(scanPos);
        }
        if (density < minDensity) {
            continue;
        }

        float preciseDistance = distance;
        uint32_t creatureId = getCreatureId(data, cell, scanPos, preciseDistance);
        uint64_t combined = static_cast<uint64_t>(preciseDistance) << 48 | static_cast<uint64_t>(density) << 40 | creatureId;
        alienAtomicMin64(&lookupResult, combined);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        if (lookupResult != 0xffffffffffffffff) {
            activity.channels[0] = 1;                                                     //something found
            activity.channels[1] = static_cast<float>((lookupResult >> 40) & 0xff) / 256;  //density
            activity.channels[2] = static_cast<float>(lookupResult >> 48) / 256;           //distance
            auto targetCreatureId = lookupResult & 0xffffffff;
            if (targetCreatureId != 0xffffffff) {
                cell->cellFunctionData.sensor.targetedCreatureId = static_cast<uint32_t>(targetCreatureId);
            }
            statistics.incNumSensorMatches(cell->color);
        } else {
            activity.channels[0] = 0;  //nothing found
        }
    }
}

__inline__ __device__ uint32_t SensorProcessor::getCreatureId(SimulationData& data, Cell* cell, float2 const& scanPos, float& preciseDistance)
{
    auto const& color = cell->cellFunctionData.sensor.restrictToColor;
    auto const& restrictToOtherMutants = cell->cellFunctionData.sensor.restrictToOtherMutants;

    //performance optimization
    if (((data.timestep + cell->age) % (6ull * 4ull)) >= 6ull) {
        return 0xffffffff;
    }
    if (cudaSimulationParameters.cellFunctionAttackerSensorDetectionFactor[cell->color] < NEAR_ZERO) {
        return 0xffffffff;
    }

    uint32_t result = 0xffffffff;
    for (float dx = -3.0f; dx < 3.0f + NEAR_ZERO; dx += 1.0f) {
        for (float dy = -3.0f; dy < 3.0f + NEAR_ZERO; dy += 1.0f) {
            auto otherCell = data.cellMap.getFirst(scanPos + float2{dx, dy});
            if (!otherCell) {
                continue;
            }
            if (cell == otherCell) {
                continue;
            }
            if (color != 255 && otherCell->color != color) {
                continue;
            }
            if (restrictToOtherMutants && otherCell->mutationId != 0 && cell->mutationId == otherCell->mutationId) {
                continue;
            }
            if (color != 255 && otherCell->color != color) {
                continue;
            }
            auto preciseDelta = data.cellMap.getCorrectedDirection(otherCell->pos - cell->pos);
            preciseDistance = Math::length(preciseDelta);
            result = otherCell->creatureId;
            if (result == cell->cellFunctionData.sensor.targetedCreatureId) {
                return result;
            }
        }
    }
    return result;
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
