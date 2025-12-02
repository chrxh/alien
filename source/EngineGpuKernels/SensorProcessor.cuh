#pragma once

#include "Object.cuh"
#include "SignalProcessor.cuh"
#include "SimulationData.cuh"

class SensorProcessor
{
public:
    __inline__ __device__ static void process(SimulationData& data, SimulationStatistics& statistics);

private:
    __inline__ __device__ static void processCell(SimulationData& data, SimulationStatistics& statistics, Cell* cell);
    __inline__ __device__ static void scanVicinityOfSensorCell(SimulationData& data, SimulationStatistics& statistics, Cell* cell);
    __inline__ __device__ static void relocateLastMatch(SimulationData& data, SimulationStatistics& statistics, Cell* cell);

    enum class ScanType
    {
        LocateMatch,
        RelocateLastMatch
    };
    __inline__ __device__ static uint64_t getMatchInfo(SimulationData& data, Cell* cell, float2 const& scanPos, float angle, float distance, ScanType scanType);

    __inline__ __device__ static uint64_t pack(float distance, float angle, float density, uint16_t misc = 0);
    __inline__ __device__ static void unpack(float& distance, float& angle, float& density, uint16_t& misc, uint64_t bytes);

    __inline__ __device__ static void writeSignal(Signal& signal, float angle, float density, float distance);

    __inline__ __device__ static uint16_t convertAngleToUint16(float angle);
    __inline__ __device__ static float convertUint16ToAngle(uint16_t b);

    static int constexpr NumScanAngles = 64;
    static float constexpr ScanStep = 8.0f;
    static int constexpr MaxNearCreatureCells = 9 * 9;
    static float constexpr RayBlockingTestLength = 10.0f;
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
        isTriggered = SignalProcessor::isAutoOrManuallyTriggered(data, cell, cell->cellTypeData.sensor.autoTriggerInterval);
        if (cell->frontAngle == VALUE_NOT_SET_FLOAT) {
            isTriggered = false;
        }
        if (isTriggered && !cell->signal.active) {
            SignalProcessor::createEmptySignal(cell);
        }
    }
    __syncthreads();

    if (isTriggered) {
        statistics.incNumSensorActivities(cell->color);

        if (cell->cellTypeData.sensor.lastMatchAvailable) {
            relocateLastMatch(data, statistics, cell);
        } else {
            scanVicinityOfSensorCell(data, statistics, cell);
        }
    }
}

__inline__ __device__ void SensorProcessor::scanVicinityOfSensorCell(SimulationData& data, SimulationStatistics& statistics, Cell* cell)
{
    __shared__ uint64_t lookupResult;

    __shared__ Cell* nearCreatureCells[MaxNearCreatureCells];
    __shared__ int numNearCreatureCells;
    __shared__ float seedAngle;

    if (threadIdx.x == 0) {
        lookupResult = 0xffffffffffffffff;
        seedAngle = data.primaryNumberGen.random(360.0f);

        data.cellMap.getMatchingCells(
            nearCreatureCells, MaxNearCreatureCells, numNearCreatureCells, cell->pos, 4.0f, cell->detached, [&](Cell* const& otherCell) {
                return cell->isSameCreature(otherCell);
            });
    }

    __syncthreads();

    auto const& densityMap = data.preprocessedSimulationData.densityMap;

    auto const startRadius = toFloat(cell->cellTypeData.sensor.minRange);
    auto endRadius = min(cudaSimulationParameters.sensorRadius.value[cell->color], toFloat(cell->cellTypeData.sensor.maxRange));

    auto angle = 360.0f * toFloat(threadIdx.x) / toFloat(blockDim.x) + seedAngle;

    // Check if ray is blocked by connections of nearby same-creature cells
    auto rayBlocked = false;
    for (int i = 0; i < numNearCreatureCells; ++i) {
        auto nearCell = nearCreatureCells[i];
        for (int j = 0, k = nearCell->numConnections; j < k; ++j) {
            auto& connectedNearCell = nearCell->connections[j].cell;
            if (Math::crossing(nearCell->pos, connectedNearCell->pos, cell->pos, cell->pos + Math::unitVectorOfAngle(angle) * RayBlockingTestLength)) {
                rayBlocked = true;
                break;
            }
        }
        if (rayBlocked) {
            break;
        }
    }

    if (!rayBlocked) {
        for (float distance = startRadius; distance <= endRadius; distance += ScanStep) {
            auto delta = Math::unitVectorOfAngle(angle) * distance;
            auto scanPos = cell->pos + delta;
            data.cellMap.correctPosition(scanPos);

            if (uint64_t matchInfo = getMatchInfo(data, cell, scanPos, angle, distance, ScanType::LocateMatch)) {
                alienAtomicMin64(&lookupResult, matchInfo);
                break;
            }

            // Block ray if it encounters structure cells
            if (densityMap.getStructureDensity(scanPos) > 0) {
                break;
            }
        }
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        if (lookupResult != 0xffffffffffffffff) {
            float distance, relAngle, density;
            uint16_t creatureIdPart;
            unpack(distance, relAngle, density, creatureIdPart, lookupResult);
            writeSignal(cell->signal, relAngle, density, distance);
            statistics.incNumSensorMatches(cell->color);

            auto refAngle = Math::angleOfVector(SignalProcessor::calcReferenceDirection(data, cell));
            auto absAngle = relAngle + refAngle + cell->frontAngle;
            auto matchPos = cell->pos + Math::unitVectorOfAngle(absAngle) * distance;
            data.cellMap.correctPosition(matchPos);

            cell->cellTypeData.sensor.lastMatchAvailable = true;
            cell->cellTypeData.sensor.lastMatch.creatureId = creatureIdPart;
            cell->cellTypeData.sensor.lastMatch.pos = matchPos;
        } else {
            cell->cellTypeData.sensor.lastMatchAvailable = false;
            cell->signal.channels[Channels::SensorFoundResult] = 0;  // Nothing found
        }
    }
}

__inline__ __device__ void SensorProcessor::relocateLastMatch(SimulationData& data, SimulationStatistics& statistics, Cell* cell)
{
        int radius = toInt(blockDim.x) / 2;
        int deltaX = toInt(threadIdx.x) - radius;
    
        __shared__ uint64_t lookupResult;
        __shared__ float refAngle;
    
        if (threadIdx.x == 0) {
            lookupResult = 0xffffffffffffffff;
            refAngle = Math::angleOfVector(SignalProcessor::calcReferenceDirection(data, cell));
        }
    
        auto centerScanPos = cell->cellTypeData.sensor.lastMatch.pos;
        for (int deltaY = -radius; deltaY < radius; ++deltaY) {

            auto delta = float2{toFloat(deltaX), toFloat(deltaY)};
            auto scanPos = centerScanPos + delta;
            auto distance = Math::length(delta);
            auto angle = Math::angleOfVector(delta);
            if (uint64_t matchInfo = getMatchInfo(data, cell, scanPos, angle, distance, ScanType::RelocateLastMatch)) {
                alienAtomicMin64(&lookupResult, matchInfo);
                break;
            }
        }
        __syncthreads();
    
        if (threadIdx.x == 0) {
            if (lookupResult != 0xffffffffffffffff) {
                float distance, relAngle, density;
                uint16_t creatureIdPart;
                unpack(distance, relAngle, density, creatureIdPart, lookupResult);
                writeSignal(cell->signal, relAngle, 0.0f, distance);
    
                statistics.incNumSensorMatches(cell->color);
    
                auto absAngle = relAngle + refAngle + cell->frontAngle;
                auto matchPos = cell->pos + Math::unitVectorOfAngle(absAngle) * distance;
                data.cellMap.correctPosition(matchPos);
    
                cell->cellTypeData.sensor.lastMatchAvailable = true;
                cell->cellTypeData.sensor.lastMatch.creatureId = creatureIdPart;
                cell->cellTypeData.sensor.lastMatch.pos = matchPos;
            } else {
                cell->cellTypeData.sensor.lastMatchAvailable = false;
                cell->signal.channels[Channels::SensorFoundResult] = 0;  // Nothing found
            }
        }
}

__inline__ __device__ uint64_t
SensorProcessor::getMatchInfo(SimulationData& data, Cell* cell, float2 const& scanPos, float angle, float distance, ScanType scanType)
{
    auto const& mode = cell->cellTypeData.sensor.mode;
    auto const& densityMap = data.preprocessedSimulationData.densityMap;

    if (mode == SensorMode_DetectEnergy) {
        auto const& minDensity = cell->cellTypeData.sensor.modeData.detectEnergy.minDensity;
        auto density = densityMap.getEnergyParticleDensity(scanPos);
        if (density >= minDensity) {
            auto refAngle = Math::angleOfVector(SignalProcessor::calcReferenceDirection(data, cell));
            return pack(distance, angle - refAngle - cell->frontAngle, density);
        }
    } else if (mode == SensorMode_DetectStructure) {
        if (densityMap.getStructureDensity(scanPos) > 0) {
            auto refAngle = Math::angleOfVector(SignalProcessor::calcReferenceDirection(data, cell));
            return pack(distance, angle - refAngle - cell->frontAngle, 1.0f);
        }
    } else if (mode == SensorMode_DetectFreeCell) {
        auto const& minDensity =  cell->cellTypeData.sensor.modeData.detectFreeCell.minDensity;
        auto const& restrictToColor = cell->cellTypeData.sensor.modeData.detectFreeCell.restrictToColor;
        auto density = densityMap.getFreeCellDensity(scanPos, restrictToColor);
        if (density >= minDensity) {
            auto refAngle = Math::angleOfVector(SignalProcessor::calcReferenceDirection(data, cell));
            return pack(distance, angle - refAngle - cell->frontAngle, density);
        }
    } else if (mode == SensorMode_DetectCreature) {
        if (scanType == ScanType::LocateMatch) {

            auto const& minNumCells = cell->cellTypeData.sensor.modeData.detectCreature.minNumCells;
            auto const& maxNumCells = cell->cellTypeData.sensor.modeData.detectCreature.maxNumCells;
            auto const& restrictToColor = cell->cellTypeData.sensor.modeData.detectCreature.restrictToColor;
            auto const& restrictToLineage = cell->cellTypeData.sensor.modeData.detectCreature.restrictToLineage;

            auto otherCell = data.cellMap.getFirst(scanPos);
            while (otherCell != nullptr) {
                // Check if this cell is part of a creature (not structure or free cell)
                if (otherCell->cellType != CellType_Structure && otherCell->cellType != CellType_Free && !cell->isSameCreature(otherCell)) {
                    bool matches = true;

                    // Apply restrictToColor filter
                    if (restrictToColor != 255 && otherCell->color != restrictToColor) {
                        matches = false;
                    }

                    // Apply minNumCells filter - if restriction is set but creature is null, fail the match
                    if (otherCell->creature == nullptr || otherCell->creature->numCells < minNumCells || otherCell->creature->numCells > maxNumCells) {
                        matches = false;
                    }

                    // Apply restrictToLineage filter - if restriction is set but either creature is null, fail the match
                    if (matches && restrictToLineage != DetectCreatureLineageRestriction_No) {
                        if (cell->creature == nullptr || otherCell->creature == nullptr) {
                            matches = false;
                        } else if (restrictToLineage == DetectCreatureLineageRestriction_SameLineage) {
                            if (cell->creature->lineageId != otherCell->creature->lineageId) {
                                matches = false;
                            }
                        } else if (restrictToLineage == DetectCreatureLineageRestriction_OtherLineage) {
                            if (cell->creature->lineageId == otherCell->creature->lineageId) {
                                matches = false;
                            }
                        }
                    }

                    if (matches) {
                        uint16_t creatureIdPart = otherCell->creature != nullptr ? static_cast<uint16_t>(otherCell->creature->id & 0xFFFF) : 0;
                        auto refAngle = Math::angleOfVector(SignalProcessor::calcReferenceDirection(data, cell));
                        return pack(distance, angle - refAngle - cell->frontAngle, 1.0f, creatureIdPart);
                    }
                }
                otherCell = otherCell->nextCell;
            }

        // Else: ScanType::Relocation
        } else {
            auto& sensor = cell->cellTypeData.sensor;
            auto otherCell = data.cellMap.getFirst(scanPos);
            while (otherCell != nullptr) {
                if (otherCell->creature != nullptr && otherCell->creature->id == sensor.lastMatch.creatureId) {
                    auto delta = data.cellMap.getCorrectedDirection(otherCell->pos - cell->pos);
                    auto distance = Math::length(delta);
                    auto angle = Math::angleOfVector(delta);
                    auto refAngle = Math::angleOfVector(SignalProcessor::calcReferenceDirection(data, cell));
                    uint16_t creatureIdPart = otherCell->creature != nullptr ? static_cast<uint16_t>(otherCell->creature->id & 0xFFFF) : 0;
                    return pack(distance, angle - refAngle - cell->frontAngle, 1.0f, creatureIdPart);
                }
                otherCell = otherCell->nextCell;
            }
        }
    }
    return 0xffffffffffffffff;
}

__inline__ __device__ uint64_t SensorProcessor::pack(float distance, float angle, float density, uint16_t misc)
{
    uint32_t angleEncoded = convertAngleToUint16(angle);
    uint32_t densityEncoded = static_cast<uint32_t>(min(65535.0f, density * 100));
    return static_cast<uint64_t>(distance) << 48 | static_cast<uint64_t>(densityEncoded) << 32 | static_cast<uint64_t>(angleEncoded) << 16
        | static_cast<uint64_t>(misc);
}

__inline__ __device__ void SensorProcessor::unpack(float& distance, float& angle, float& density, uint16_t& misc, uint64_t bytes)
{
    distance = toFloat(bytes >> 48);
    density = toFloat((bytes >> 32) & 0xFFFF) / 100;
    angle = convertUint16ToAngle(static_cast<int16_t>((bytes >> 16) & 0xFFFF));
    misc = static_cast<int16_t>(bytes & 0xFFFF);
}

__inline__ __device__ void SensorProcessor::writeSignal(Signal& signal, float angle, float density, float distance)
{
    signal.channels[Channels::SensorFoundResult] = 1;               // Something found
    signal.channels[Channels::SensorAngle] = angle / 180.0f;                       // Angle: between -1.0 and 1.0
    signal.channels[Channels::SensorDensity] = min(1.0f, density);  // Normalized density (1.0 = 64 cells in 8x8 region)
    signal.channels[Channels::SensorDistance] = 1.0f - min(1.0f, distance / 256);  // Distance: 1 = close, 0 = far away
}

__inline__ __device__ uint16_t SensorProcessor::convertAngleToUint16(float angle)
{
    angle = Math::getNormalizedAngle(angle, -180.0f);
    int result = static_cast<int>(angle / 180.0f * 32768.0f);
    return static_cast<uint16_t>(result);
}

__inline__ __device__ float SensorProcessor::convertUint16ToAngle(uint16_t b)
{
    //0 to 32767 => 0 to 179 degree
    //32768 to 65535 => -179 to 0 degree
    if (b < 32768) {
        return (0.5f + static_cast<float>(b)) * (180.0f / 32768.0f);
    } else {
        return (-65536.0f - 0.5f + static_cast<float>(b)) * (180.0f / 32768.0f);
    }
}

//__inline__ __device__ uint32_t SensorProcessor::getCellDensity(
//    uint64_t const& timestep,
//    Cell* const& cell,
//    uint8_t const& restrictToColor,
//    SensorRestrictToCreatures const& restrictToCreatures,
//    DensityMap const& densityMap,
//    float2 const& scanPos)
//{
//    uint32_t result;
//    if (restrictToCreatures == SensorRestrictToCreatures_NoRestriction) {
//        if (restrictToColor == 255) {
//            result = densityMap.getCellDensity(scanPos);
//        }
//        if (restrictToColor != 255) {
//            result = densityMap.getColorDensity(scanPos, restrictToColor);
//        }
//    } else {
//        if (restrictToCreatures == SensorRestrictToCreatures_RestrictToSameMutants) {
//            if (cell->creature != nullptr) {
//                result = densityMap.getSameMutantDensity(scanPos, cell->creature->lineageId);
//            } else {
//                result = 0;
//            }
//        }
//        if (restrictToCreatures == SensorRestrictToCreatures_RestrictToOtherMutants) {
//            if (cell->creature != nullptr) {
//                result = densityMap.getOtherMutantDensity(timestep, scanPos, cell->creature->lineageId);
//            } else {
//                result = 0;
//            }
//        }
//        if (restrictToCreatures == SensorRestrictToCreatures_RestrictToFreeCells) {
//            result = densityMap.getFreeCellDensity(scanPos);
//        }
//        if (restrictToCreatures == SensorRestrictToCreatures_RestrictToStructures) {
//            result = densityMap.getStructureDensity(scanPos);
//        }
//        if (restrictToCreatures == SensorRestrictToCreatures_RestrictToLessComplexMutants) {
//            if (cell->creature != nullptr) {
//                result = densityMap.getLessComplexMutantDensity(scanPos, cell->creature->numCells);
//            } else {
//                result = 0;
//            }
//        }
//        if (restrictToCreatures == SensorRestrictToCreatures_RestrictToMoreComplexMutants) {
//            if (cell->creature != nullptr)
//                result = densityMap.getMoreComplexMutantDensity(scanPos, cell->creature->numCells);
//            else {
//                result = 0;
//            }
//        }
//        if (restrictToColor != 255) {
//            result = min(result, densityMap.getColorDensity(scanPos, restrictToColor));
//        }
//    }
//    return result;
//}
//
//__inline__ __device__ void SensorProcessor::searchNeighborhood(SimulationData& data, SimulationStatistics& statistics, Cell* cell)
//{
//    __shared__ uint32_t minDensity;
//    __shared__ uint8_t restrictToColor;
//    __shared__ SensorRestrictToCreatures restrictToCreatures;
//    __shared__ float refAngle;
//    __shared__ uint64_t lookupResult;
//    __shared__ bool rayBlocked[NumScanAngles];
//
//    __shared__ Cell* nearCreatureCells[9 * 9];
//    __shared__ int numNearCreatureCells;
//
//    if (threadIdx.x == 0) {
//        refAngle = Math::angleOfVector(SignalProcessor::calcReferenceDirection(data, cell));
//        minDensity = toInt(cell->cellTypeData.sensor.minDensity * 64 + 0.5f);
//        restrictToColor = cell->cellTypeData.sensor.restrictToColor;
//        restrictToCreatures = cell->cellTypeData.sensor.restrictToCreatures;
//        lookupResult = 0xffffffffffffffff;
//
//        data.cellMap.getMatchingCells(nearCreatureCells, 9 * 9, numNearCreatureCells, cell->pos, 4.0f, cell->detached, [&](Cell* const& otherCell) {
//            return cell->isSameCreature(otherCell);
//        });
//    }
//    auto const angleIndex = threadIdx.x;
//
//    __syncthreads();
//
//    auto const& densityMap = data.preprocessedSimulationData.densityMap;
//
//    auto const startRadius = max(toFloat(cell->cellTypeData.sensor.minRange), calcStartDistanceForScanning(restrictToColor, restrictToCreatures, cell->color));
//    auto endRadius = cudaSimulationParameters.sensorRadius.value[cell->color];
//    if (cell->cellTypeData.sensor.maxRange >= 0) {
//        endRadius = min(endRadius, toFloat(cell->cellTypeData.sensor.maxRange));
//    }
//
//    float angle = 360.0f / NumScanAngles * toFloat(angleIndex);
//
//    rayBlocked[angleIndex] = false;
//    for (int i = 0; i < numNearCreatureCells; ++i) {
//        auto nearCell = nearCreatureCells[i];
//        for (int j = 0, k = nearCell->numConnections; j < k; ++j) {
//            auto& connectedNearCell = nearCell->connections[j].cell;
//            if (Math::crossing(nearCell->pos, connectedNearCell->pos, cell->pos, cell->pos + Math::unitVectorOfAngle(angle) * 10.0f)) {
//                rayBlocked[angleIndex] = true;
//            }
//        }
//    }
//    for (float radius = startRadius; radius <= endRadius; radius += ScanStep) {
//        if (!rayBlocked[angleIndex]) {
//
//            auto delta = Math::unitVectorOfAngle(angle) * radius;
//            auto scanPos = cell->pos + delta;
//            data.cellMap.correctPosition(scanPos);
//
//            uint32_t density = 0;
//
//            if (restrictToCreatures == SensorRestrictToCreatures_NoRestriction || restrictToCreatures == SensorRestrictToCreatures_RestrictToStructures
//                || densityMap.getStructureDensity(scanPos) == 0) {
//                density = getCellDensity(data.timestep, cell, restrictToColor, restrictToCreatures, densityMap, scanPos);
//            } else {
//                rayBlocked[angleIndex] = true;
//            }
//            if (density >= minDensity) {
//                float preciseDistance = radius;
//                uint32_t relAngleEncoded = convertAngleToData(angle - refAngle - cell->frontAngle);
//                uint64_t combined =
//                    static_cast<uint64_t>(preciseDistance) << 48 | static_cast<uint64_t>(density) << 40 | static_cast<uint64_t>(relAngleEncoded) << 32;
//                alienAtomicMin64(&lookupResult, combined);
//            }
//        }
//        __syncthreads();
//    }
//
//    if (threadIdx.x == 0) {
//        if (lookupResult != 0xffffffffffffffff) {
//
//            auto relAngle = convertDataToAngle(static_cast<int8_t>((lookupResult >> 32) & 0xff));
//            auto distance = toFloat(lookupResult >> 48);
//            auto absAngle = relAngle + refAngle + cell->frontAngle;
//            auto scanPos = cell->pos + Math::unitVectorOfAngle(absAngle) * distance;
//            flagDetectedCells(data, cell, scanPos);
//
//            cell->signal.channels[Channels::SensorFoundResult] = 1;                                      //something found
//            cell->signal.channels[Channels::SensorAngle] = relAngle / 180.0f;                            //angle: between -1.0 and 1.0
//            cell->signal.channels[Channels::SensorDensity] = toFloat((lookupResult >> 40) & 0xff) / 64;  //density
//
//            cell->signal.channels[Channels::SensorDistance] = 1.0f - min(1.0f, distance / 256);  //distance: 1 = close, 0 = far away
//            statistics.incNumSensorMatches(cell->color);
//        } else {
//            cell->signal.channels[Channels::SensorFoundResult] = 0;  //nothing found
//        }
//    }
//    __syncthreads();
//}
//
//__inline__ __device__ void SensorProcessor::flagDetectedCells(SimulationData& data, Cell* cell, float2 const& scanPos)
//{
//    if (cell->creature == nullptr) {
//        return;
//    }
//
//    auto const& restrictToColor = cell->cellTypeData.sensor.restrictToColor;
//    auto const& restrictToCreatures = cell->cellTypeData.sensor.restrictToCreatures;
//
//    for (float dx = -3.0f; dx < 3.0f + NEAR_ZERO; dx += 1.0f) {
//        for (float dy = -3.0f; dy < 3.0f + NEAR_ZERO; dy += 1.0f) {
//            auto otherCell = data.cellMap.getFirst(scanPos + float2{dx, dy});
//            if (!otherCell) {
//                continue;
//            }
//            if (cell == otherCell) {
//                continue;
//            }
//            if (restrictToColor != 255 && otherCell->color != restrictToColor) {
//                continue;
//            }
//            if (restrictToCreatures == SensorRestrictToCreatures_RestrictToSameMutants
//                || restrictToCreatures == SensorRestrictToCreatures_RestrictToOtherMutants
//                || restrictToCreatures == SensorRestrictToCreatures_RestrictToLessComplexMutants
//                || restrictToCreatures == SensorRestrictToCreatures_RestrictToMoreComplexMutants) {
//                if (otherCell->cellType == CellType_Free || otherCell->cellType == CellType_Structure) {
//                    continue;
//                }
//            }
//            if (restrictToCreatures == SensorRestrictToCreatures_RestrictToSameMutants
//                && (cell->creature == nullptr || otherCell->creature == nullptr || cell->creature->lineageId != otherCell->creature->lineageId)) {
//                continue;
//            }
//            if (restrictToCreatures == SensorRestrictToCreatures_RestrictToOtherMutants
//                && (cell->creature == nullptr || otherCell->creature == nullptr || cell->creature->lineageId == otherCell->creature->lineageId
//                    || cell->creature->lineageId == otherCell->creature->ancestorId)) {
//                continue;
//            }
//            if (restrictToCreatures == SensorRestrictToCreatures_RestrictToFreeCells && otherCell->cellType != CellType_Free) {
//                continue;
//            }
//            if (restrictToCreatures == SensorRestrictToCreatures_RestrictToStructures && otherCell->cellType != CellType_Structure) {
//                continue;
//            }
//            if (restrictToCreatures == SensorRestrictToCreatures_RestrictToLessComplexMutants
//                && (cell->creature == nullptr || otherCell->creature == nullptr || otherCell->creature->numCells >= cell->creature->numCells)) {
//                continue;
//            }
//            if (restrictToCreatures == SensorRestrictToCreatures_RestrictToMoreComplexMutants
//                && (cell->creature == nullptr || otherCell->creature == nullptr || otherCell->creature->numCells <= cell->creature->numCells)) {
//                continue;
//            }
//
//            otherCell->detectedByCreatureId = static_cast<uint16_t>(cell->creature->id & 0xffff);
//        }
//    }
//}
//
//__inline__ __device__ float
//SensorProcessor::calcStartDistanceForScanning(uint8_t const& restrictToColor, SensorRestrictToCreatures const& restrictToCreatures, int const& color)
//{
//    return (restrictToColor == 255 || restrictToColor == color)
//            && (restrictToCreatures == SensorRestrictToCreatures_NoRestriction || restrictToCreatures == SensorRestrictToCreatures_RestrictToSameMutants)
//        ? 14.0f
//        : 0.0f;
//}
//
//__inline__ __device__ uint8_t SensorProcessor::convertAngleToData(float angle)
//{
//    angle = Math::getNormalizedAngle(angle, -180.0f);
//    int result = static_cast<int>(angle * 128.0f / 180.0f);
//    return static_cast<uint8_t>(result);
//}
//
//__inline__ __device__ float SensorProcessor::convertDataToAngle(uint8_t b)
//{
//    //0 to 127 => 0 to 179 degree
//    //128 to 255 => -179 to 0 degree
//    if (b < 128) {
//        return (0.5f + static_cast<float>(b)) * (180.0f / 128.0f);
//    } else {
//        return (-256.0f - 0.5f + static_cast<float>(b)) * (180.0f / 128.0f);
//    }
//}
