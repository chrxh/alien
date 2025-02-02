#pragma once

#include "Object.cuh"
#include "SimulationData.cuh"
#include "SignalProcessor.cuh"

class SensorProcessor
{
public:
    __inline__ __device__ static void process(SimulationData& data, SimulationStatistics& statistics);

private:
    __inline__ __device__ static void processCell(SimulationData& data, SimulationStatistics& statistics, Cell* cell);

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

    static int constexpr NumScanAngles = 64;
    static int constexpr NumScanPoints = 64;
    static float constexpr ScanStep = 8.0f;
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
    __shared__ bool isTriggered;
    if (threadIdx.x == 0) {
        isTriggered = SignalProcessor::isTriggeredAndCreateSignalIfTriggered(data, cell, cell->cellTypeData.sensor.autoTriggerInterval);
    }
    __syncthreads();

    if (isTriggered) {
        statistics.incNumSensorActivities(cell->color);
        searchNeighborhood(data, statistics, cell);
        cell->signal.origin = SignalOrigin_Sensor;
    }
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
    __shared__ float refAngle;
    __shared__ uint64_t lookupResult;
    __shared__ bool rayBlocked[NumScanAngles];

    __shared__ Cell* nearCreatureCells[9 * 9];
    __shared__ int numNearCreatureCells;

    if (threadIdx.x == 0) {
        refAngle = Math::angleOfVector(SignalProcessor::calcReferenceDirection(data, cell));
        minDensity = toInt(cell->cellTypeData.sensor.minDensity * 64 + 0.5f);
        restrictToColor = cell->cellTypeData.sensor.restrictToColor;
        restrictToMutants = cell->cellTypeData.sensor.restrictToMutants;
        lookupResult = 0xffffffffffffffff;

        data.cellMap.getMatchingCells(
            nearCreatureCells, 9 * 9,
            numNearCreatureCells,
            cell->pos,
            4.0f,
            cell->detached,
            [&](Cell* const& otherCell) { return cell->creatureId == otherCell->creatureId; });
    }
    auto const angleIndex = threadIdx.x;

    __syncthreads();

    auto const& densityMap = data.preprocessedSimulationData.densityMap;

    auto const startRadius = max(toFloat(cell->cellTypeData.sensor.minRange), calcStartDistanceForScanning(restrictToColor, restrictToMutants, cell->color));
    auto endRadius = cudaSimulationParameters.cellTypeSensorRange[cell->color];
    if (cell->cellTypeData.sensor.maxRange >= 0) {
        endRadius = min(endRadius, toFloat(cell->cellTypeData.sensor.maxRange));
    }

    float angle = 360.0f / NumScanAngles * toFloat(angleIndex);

    rayBlocked[angleIndex] = false;
    for (int i = 0; i < numNearCreatureCells; ++i) {
        auto nearCell = nearCreatureCells[i];
        for (int j = 0, k = nearCell->numConnections; j < k; ++j) {
            auto& connectedNearCell = nearCell->connections[j].cell;
            if (Math::crossing(nearCell->pos, connectedNearCell->pos, cell->pos, cell->pos + Math::unitVectorOfAngle(angle) * 10.0f)) {
                rayBlocked[angleIndex] = true;
            }
        }
    }
    for (float radius = startRadius; radius <= endRadius; radius += ScanStep) {
        if (!rayBlocked[angleIndex]) {

            auto delta = Math::unitVectorOfAngle(angle) * radius;
            auto scanPos = cell->pos + delta;
            data.cellMap.correctPosition(scanPos);

            uint32_t density = 0;

            if (restrictToMutants == SensorRestrictToMutants_NoRestriction || restrictToMutants == SensorRestrictToMutants_RestrictToStructures
                || densityMap.getStructureDensity(scanPos) == 0) {
                density = getCellDensity(data.timestep, cell, restrictToColor, restrictToMutants, densityMap, scanPos);
            } else {
                rayBlocked[angleIndex] = true;
            }
            if (density >= minDensity) {
                float preciseDistance = radius;
                uint32_t relAngleEncoded = convertAngleToData(Math::subtractAngle(angle, refAngle));
                uint64_t combined =
                    static_cast<uint64_t>(preciseDistance) << 48 | static_cast<uint64_t>(density) << 40 | static_cast<uint64_t>(relAngleEncoded) << 32;
                alienAtomicMin64(&lookupResult, combined);
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        if (lookupResult != 0xffffffffffffffff) {

            auto relAngle = convertDataToAngle(static_cast<int8_t>((lookupResult >> 32) & 0xff));
            auto distance = toFloat(lookupResult >> 48);
            auto angle = refAngle + relAngle;
            auto scanPos = cell->pos + Math::unitVectorOfAngle(angle) * distance;
            flagDetectedCells(data, cell, scanPos);

            cell->signal.channels[0] = 1;                                    //something found
            cell->signal.channels[1] = relAngle / 360.0f;                          //angle: between -0.5 and 0.5
            cell->signal.channels[2] = toFloat((lookupResult >> 40) & 0xff) / 64;  //density

            cell->signal.channels[3] = 1.0f - min(1.0f, distance / 256);  //distance: 1 = close, 0 = far away
            statistics.incNumSensorMatches(cell->color);
            auto delta = data.cellMap.getCorrectedDirection(scanPos - cell->pos);
            cell->signal.targetX = delta.x;
            cell->signal.targetY = delta.y;
        } else {
            cell->signal.channels[0] = 0;  //nothing found
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
            if (restrictToMutants == SensorRestrictToMutants_RestrictToSameMutants || restrictToMutants == SensorRestrictToMutants_RestrictToOtherMutants
                || restrictToMutants == SensorRestrictToMutants_RestrictToLessComplexMutants
                || restrictToMutants == SensorRestrictToMutants_RestrictToMoreComplexMutants) {
                if (otherCell->cellType == CellType_Free || otherCell->cellType == CellType_Structure) {
                    continue;
                }
            }
            if (restrictToMutants == SensorRestrictToMutants_RestrictToSameMutants && cell->mutationId != otherCell->mutationId) {
                continue;
            }
            if (restrictToMutants == SensorRestrictToMutants_RestrictToOtherMutants
                && (cell->mutationId == otherCell->mutationId || static_cast<uint8_t>(cell->mutationId & 0xff) == otherCell->ancestorMutationId)) {
                continue;
            }
            if (restrictToMutants == SensorRestrictToMutants_RestrictToFreeCells && otherCell->cellType != CellType_Free) {
                continue;
            }
            if (restrictToMutants == SensorRestrictToMutants_RestrictToStructures && otherCell->cellType != CellType_Structure) {
                continue;
            }
            if (restrictToMutants == SensorRestrictToMutants_RestrictToLessComplexMutants && otherCell->genomeComplexity >= cell->genomeComplexity) {
                continue;
            }
            if (restrictToMutants == SensorRestrictToMutants_RestrictToMoreComplexMutants && otherCell->genomeComplexity <= cell->genomeComplexity) {
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
    angle = Math::normalizedAngle(angle, -180.0f);
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
