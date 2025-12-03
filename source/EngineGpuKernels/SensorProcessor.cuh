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

    auto seedDistance = data.primaryNumberGen.random(0, ScanStep);
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
        for (float distance = seedDistance; distance <= endRadius; distance += ScanStep) {
            auto delta = Math::unitVectorOfAngle(angle) * distance;
            auto scanPos = cell->pos + delta;
            data.cellMap.correctPosition(scanPos);

            if (distance > startRadius) {
                uint64_t matchInfo = getMatchInfo(data, cell, scanPos, angle, distance, ScanType::LocateMatch);
                if (matchInfo != 0xffffffffffffffff) {
                    alienAtomicMin64(&lookupResult, matchInfo);
                    break;
                }
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

            // No relocation for structures 
            if (cell->cellTypeData.sensor.mode != SensorMode_DetectStructure) {
                cell->cellTypeData.sensor.lastMatchAvailable = true;
                cell->cellTypeData.sensor.lastMatch.creatureId = creatureIdPart;
                cell->cellTypeData.sensor.lastMatch.pos = matchPos;
            }
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
            uint64_t matchInfo = getMatchInfo(data, cell, scanPos, angle, distance, ScanType::RelocateLastMatch);
            if (matchInfo != 0xffffffffffffffff) {
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

                    if (restrictToColor != 255 && otherCell->color != restrictToColor) {
                        matches = false;
                    }
                    if (otherCell->creature == nullptr) {
                        matches = false;
                    }
                    if (minNumCells > 0 && otherCell->creature->numCells < minNumCells) {
                        matches = false;
                    }
                    if (maxNumCells > 0 && otherCell->creature->numCells > maxNumCells) {
                        matches = false;
                    }
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
                if (otherCell->creature != nullptr && (otherCell->creature->id & 0xffff) == sensor.lastMatch.creatureId) {
                    auto delta = data.cellMap.getCorrectedDirection(otherCell->pos - cell->pos);
                    auto distance = Math::length(delta);
                    auto angle = Math::angleOfVector(delta);
                    auto refAngle = Math::angleOfVector(SignalProcessor::calcReferenceDirection(data, cell));
                    uint16_t creatureIdPart = otherCell->creature != nullptr ? static_cast<uint16_t>(otherCell->creature->id & 0xffff) : 0;
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
