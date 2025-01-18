#pragma once

#include "Object.cuh"
#include "SimulationData.cuh"
#include "SignalProcessor.cuh"

class SensorProcessor
{
public:
    __inline__ __device__ static void process(SimulationData& data, SimulationStatistics& statistics);

private:
    static int constexpr NumScanAngles = 64;
    static int constexpr NumScanPoints = 64;
    static float constexpr ScanStep = 8.0f;

    __inline__ __device__ static void processCell(SimulationData& data, SimulationStatistics& statistics, Cell* cell);

    __inline__ __device__ static bool isTriggeredAndCreateSignalIfTriggered(SimulationData& data, Cell* cell);

    __inline__ __device__ static uint32_t getCellDensity(
        uint64_t const& timestep,
        Cell* const& cell,
        uint8_t const& restrictToColor,
        SensorRestrictToMutants const& restrictToMutants,
        DensityMap const& densityMap,
        float2 const& scanPos);
    __inline__ __device__ static void searchNeighborhood(SimulationData& data, SimulationStatistics& statistics, Cell* cell);

    __inline__ __device__ static void flagDetectedCells(SimulationData& data, Cell* cell, float2 const& scanPos);

    __inline__ __device__ static float
    calcStartDistanceForScanning(uint8_t const& restrictToColor, SensorRestrictToMutants const& restrictToMutants, int const& color);

    __inline__ __device__ static uint8_t convertAngleToData(float angle);
    __inline__ __device__ static float convertDataToAngle(uint8_t b);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__inline__ __device__ void SensorProcessor::process(SimulationData& data, SimulationStatistics& statistics)
{
    auto& operations = data.cellTypeOperations[CellType_Sensor];
    auto partition = calcBlockPartition(operations.getNumEntries());
    for (int i = partition.startIndex; i <= partition.endIndex; ++i) {
        processCell(data, statistics, operations.at(i).cell);
    }
}

__inline__ __device__ void SensorProcessor::processCell(SimulationData& data, SimulationStatistics& statistics, Cell* cell)
{
    if (isTriggeredAndCreateSignalIfTriggered(data, cell)) {
        statistics.incNumSensorActivities(cell->color);
        searchNeighborhood(data, statistics, cell);
        cell->signal.origin = SignalOrigin_Sensor;
    }
}

__inline__ __device__ bool SensorProcessor::isTriggeredAndCreateSignalIfTriggered(SimulationData& data, Cell* cell)
{
    if (cell->cellTypeData.sensor.autoTriggerInterval == 0) {
        if (!cell->signal.active) {
            return false;
        }
        if (cell->signal.active && abs(cell->signal.channels[0]) < cudaSimulationParameters.cellTypeSensorSignalThreshold) {
            return false;
        }
    } else {
        if (!cell->signal.active) {
            SignalProcessor::createEmptySignal(cell);
        }
        auto activationTime = max(MAX_SIGNAL_RELAXATION_TIME + 1, cell->cellTypeData.sensor.autoTriggerInterval);
        if (data.timestep % activationTime != 0) {
            return false;
        }
    }
    return true;
}

__inline__ __device__ uint32_t SensorProcessor::getCellDensity(
    uint64_t const& timestep,
    Cell* const& cell,
    uint8_t const& restrictToColor,
    SensorRestrictToMutants const& restrictToMutants,
    DensityMap const& densityMap,
    float2 const& scanPos)
{
    uint32_t result;
    if (restrictToMutants == SensorRestrictToMutants_NoRestriction) {
        if (restrictToColor == 255) {
            result = densityMap.getCellDensity(scanPos);
        }
        if (restrictToColor != 255) {
            result = densityMap.getColorDensity(scanPos, restrictToColor);
        }
    } else {
        if (restrictToMutants == SensorRestrictToMutants_RestrictToSameMutants) {
            result = densityMap.getSameMutantDensity(scanPos, cell->mutationId);
        }
        if (restrictToMutants == SensorRestrictToMutants_RestrictToOtherMutants) {
            result = densityMap.getOtherMutantDensity(timestep, scanPos, cell->mutationId);
        }
        if (restrictToMutants == SensorRestrictToMutants_RestrictToFreeCells) {
            result = densityMap.getFreeCellDensity(scanPos);
        }
        if (restrictToMutants == SensorRestrictToMutants_RestrictToStructures) {
            result = densityMap.getStructureDensity(scanPos);
        }
        if (restrictToMutants == SensorRestrictToMutants_RestrictToLessComplexMutants) {
            result = densityMap.getLessComplexMutantDensity(scanPos, cell->genomeComplexity);
        }
        if (restrictToMutants == SensorRestrictToMutants_RestrictToMoreComplexMutants) {
            result = densityMap.getMoreComplexMutantDensity(scanPos, cell->genomeComplexity);
        }
        if (restrictToColor != 255) {
            result = min(result, densityMap.getColorDensity(scanPos, restrictToColor));
        }
    }
    return result;
}

__inline__ __device__ void
SensorProcessor::searchNeighborhood(SimulationData& data, SimulationStatistics& statistics, Cell* cell)
{
    __shared__ uint32_t minDensity;
    __shared__ uint8_t restrictToColor;
    __shared__ SensorRestrictToMutants restrictToMutants;
    __shared__ float refScanAngle;
    __shared__ uint64_t lookupResult;
    __shared__ bool blockedByWall[NumScanAngles];
    __shared__ int8_t minRange;

    if (threadIdx.x == 0) {
        refScanAngle = Math::angleOfVector(SignalProcessor::calcSignalDirection(data, cell));
        minDensity = toInt(cell->cellTypeData.sensor.minDensity * 64);
        minRange = cell->cellTypeData.sensor.minRange;
        restrictToColor = cell->cellTypeData.sensor.restrictToColor;
        restrictToMutants = cell->cellTypeData.sensor.restrictToMutants;
        lookupResult = 0xffffffffffffffff;
    }
    __syncthreads();
    auto const partition = calcPartition(NumScanAngles, threadIdx.x, blockDim.x);
    for (int angleIndex = partition.startIndex; angleIndex <= partition.endIndex; ++angleIndex) {
        blockedByWall[angleIndex] = false;
    }
    __syncthreads();

    auto const startRadius = calcStartDistanceForScanning(restrictToColor, restrictToMutants, cell->color);
    auto const& densityMap = data.preprocessedSimulationData.densityMap;

    auto maxRange = cudaSimulationParameters.cellTypeSensorRange[cell->color];
    if (cell->cellTypeData.sensor.maxRange >= 0) {
        maxRange = min(maxRange, toFloat(cell->cellTypeData.sensor.maxRange));
    }
    for (float radius = startRadius; radius <= maxRange; radius += ScanStep) {
        if (minRange < 0 || minRange <= radius) {
            for (int angleIndex = partition.startIndex; angleIndex <= partition.endIndex; ++angleIndex) {
                float angle = 360.0f / NumScanAngles * angleIndex;

                auto delta = Math::unitVectorOfAngle(angle) * radius;
                auto scanPos = cell->pos + delta;
                data.cellMap.correctPosition(scanPos);

                uint32_t density = 0;
                if (!blockedByWall[angleIndex]) {
                    if (restrictToMutants == SensorRestrictToMutants_NoRestriction || restrictToMutants == SensorRestrictToMutants_RestrictToStructures
                        || densityMap.getStructureDensity(scanPos) == 0) {
                        density = getCellDensity(data.timestep, cell, restrictToColor, restrictToMutants, densityMap, scanPos);
                    } else {
                        blockedByWall[angleIndex] = true;
                    }
                }
                if (density < minDensity) {
                    continue;
                }
                float preciseAngle = angle;
                float preciseDistance = radius;
                auto relAngle = Math::subtractAngle(preciseAngle, refScanAngle);
                uint32_t relAngleData = convertAngleToData(relAngle);
                uint64_t combined =
                    static_cast<uint64_t>(preciseDistance) << 48 | static_cast<uint64_t>(density) << 40 | static_cast<uint64_t>(relAngleData) << 32;
                alienAtomicMin64(&lookupResult, combined);
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        if (lookupResult != 0xffffffffffffffff) {

            auto angle = convertDataToAngle(static_cast<int8_t>((lookupResult >> 32) & 0xff));
            auto distance = toFloat(lookupResult >> 48);
            auto scanAngle = refScanAngle + angle;
            auto scanPos = cell->pos + Math::unitVectorOfAngle(scanAngle) * distance;
            flagDetectedCells(data, cell, scanPos);

            cell->signal.channels[0] = 1;                                    //something found
            cell->signal.channels[1] = toFloat((lookupResult >> 40) & 0xff) / 64;  //density

            cell->signal.channels[2] = 1.0f - min(1.0f, distance / 256);  //distance: 1 = close, 0 = far away

            auto movementTowardTargetedObject = !cudaSimulationParameters.cellTypeMuscleMovementTowardTargetedObject;
            cell->signal.channels[3] = movementTowardTargetedObject ? angle / 360.0f : 0;  //angle: between -0.5 and 0.5
            cell->cellTypeData.sensor.memoryChannel1 = cell->signal.channels[1];
            cell->cellTypeData.sensor.memoryChannel2 = cell->signal.channels[2];
            cell->cellTypeData.sensor.memoryChannel3 = cell->signal.channels[3];
            statistics.incNumSensorMatches(cell->color);
            auto delta = data.cellMap.getCorrectedDirection(scanPos - cell->pos);
            cell->signal.targetX = delta.x;
            cell->signal.targetY = delta.y;
            cell->cellTypeData.sensor.memoryTargetX = delta.x;
            cell->cellTypeData.sensor.memoryTargetY = delta.y;
        } else {
            cell->signal.channels[0] = 0;  //nothing found
            //signal.channels[1] = cell->cellTypeData.sensor.memoryChannel1;
            //signal.channels[2] = cell->cellTypeData.sensor.memoryChannel2;
            //signal.channels[3] = cell->cellTypeData.sensor.memoryChannel3;
            //signal.targetX = cell->cellTypeData.sensor.memoryTargetX;
            //signal.targetY = cell->cellTypeData.sensor.memoryTargetY;
            cell->signal.targetX = 0;
            cell->signal.targetY = 0;
        }
    }
    __syncthreads();
}

__inline__ __device__ void SensorProcessor::flagDetectedCells(SimulationData& data, Cell* cell, float2 const& scanPos)
{
    auto const& restrictToColor = cell->cellTypeData.sensor.restrictToColor;
    auto const& restrictToMutants = cell->cellTypeData.sensor.restrictToMutants;

    for (float dx = -3.0f; dx < 3.0f + NEAR_ZERO; dx += 1.0f) {
        for (float dy = -3.0f; dy < 3.0f + NEAR_ZERO; dy += 1.0f) {
            auto otherCell = data.cellMap.getFirst(scanPos + float2{dx, dy});
            if (!otherCell) {
                continue;
            }
            if (cell == otherCell) {
                continue;
            }
            if (restrictToColor != 255 && otherCell->color != restrictToColor) {
                continue;
            }
            if (restrictToMutants == SensorRestrictToMutants_RestrictToSameMutants && cell->mutationId != otherCell->mutationId) {
                continue;
            }
            if (restrictToMutants == SensorRestrictToMutants_RestrictToOtherMutants
                && (cell->mutationId == otherCell->mutationId || otherCell->cellType == CellType_Free || otherCell->cellType == CellType_Structure
                    || static_cast<uint8_t>(cell->mutationId & 0xff) == otherCell->ancestorMutationId)) {
                continue;
            }
            if (restrictToMutants == SensorRestrictToMutants_RestrictToFreeCells && otherCell->cellType != CellType_Free) {
                continue;
            }
            if (restrictToMutants == SensorRestrictToMutants_RestrictToStructures && otherCell->cellType != CellType_Structure) {
                continue;
            }
            if (restrictToMutants == SensorRestrictToMutants_RestrictToLessComplexMutants
                && (otherCell->genomeComplexity >= cell->genomeComplexity || otherCell->cellType == CellType_Free
                    || otherCell->cellType == CellType_Structure)) {
                continue;
            }
            if (restrictToMutants == SensorRestrictToMutants_RestrictToMoreComplexMutants
                && (otherCell->genomeComplexity <= cell->genomeComplexity || otherCell->cellType == CellType_Free
                    || otherCell->cellType == CellType_Structure)) {
                continue;
            }

            otherCell->detectedByCreatureId = static_cast<uint16_t>(cell->creatureId & 0xffff);
        }
    }
}

__inline__ __device__ float
SensorProcessor::calcStartDistanceForScanning(uint8_t const& restrictToColor, SensorRestrictToMutants const& restrictToMutants, int const& color)
{
    return (restrictToColor == 255 || restrictToColor == color)
            && (restrictToMutants == SensorRestrictToMutants_NoRestriction || restrictToMutants == SensorRestrictToMutants_RestrictToSameMutants)
        ? 14.0f
        : 0.0f;
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
