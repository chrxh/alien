#pragma once

#include "Entities.cuh"
#include "SignalProcessor.cuh"
#include "SimulationData.cuh"

class SensorProcessor
{
public:
    __inline__ __device__ static void process(SimulationData& data, SimulationStatistics& statistics);

private:
    __inline__ __device__ static void processCell(SimulationData& data, SimulationStatistics& statistics, Object* object);

    __inline__ __device__ static void processDetection(SimulationData& data, SimulationStatistics& statistics, Object* object);
    __inline__ __device__ static void processTelemetry(SimulationData& data, SimulationStatistics& statistics, Object* object);
    __inline__ __device__ static void initialScan(SimulationData& data, SimulationStatistics& statistics, Object* object);
    __inline__ __device__ static void relocateLastMatch(SimulationData& data, SimulationStatistics& statistics, Object* object);

    enum class ScanType
    {
        LocateMatch,
        RelocateLastMatch
    };
    __inline__ __device__ static uint64_t
    getMatchInfo(SimulationData& data, Object* object, float2 const& scanPos, float absAngle, float distance, ScanType scanType);

    __inline__ __device__ static bool
    isRayBlockedByCreatureConnections(Object** nearSameCreatureCells, int numNearSameCreatureCells, float2 const& rayOrigin, float angle);

    __inline__ __device__ static uint64_t pack(float distance, float angle, float density, uint16_t misc = 0);
    __inline__ __device__ static void unpack(float& distance, float& angle, float& density, uint16_t& misc, uint64_t bytes);

    __inline__ __device__ static void writeSignal(Signal& signal, float angle, float density, float distance);

    __inline__ __device__ static uint16_t convertAngleToUint16(float angle);
    __inline__ __device__ static float convertUint16ToAngle(uint16_t b);

    __inline__ __device__ static float calcCreatureDensityFromNumCells(uint32_t numCells);

    static int constexpr NumScanRays = 64;
    static int constexpr RelocationSearchRadius = 32;  // Search grid is (2*radius) x (2*radius) = 64x64
    static float constexpr ScanStep = 8.0f;
    static int constexpr MaxSameNearCreatureCells = 9 * 9;
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
        processCell(data, statistics, operations.at(i).object);
    }
}

__inline__ __device__ void SensorProcessor::processCell(SimulationData& data, SimulationStatistics& statistics, Object* object)
{
    __shared__ bool isTriggered;
    if (threadIdx.x == 0) {
        isTriggered = SignalProcessor::isAutoOrManuallyTriggered(data, object, object->typeData.cell.cellTypeData.sensor.autoTriggerInterval);
        if (object->typeData.cell.frontAngle == VALUE_NOT_SET_FLOAT) {
            isTriggered = false;
        }
    }
    __syncthreads();

    if (isTriggered) {
        statistics.incNumSensorActivities(object->color);

        if (object->typeData.cell.cellTypeData.sensor.mode != SensorMode_Telemetry) {
            processDetection(data, statistics, object);
        } else {
            processTelemetry(data, statistics, object);
        }
    }
}

__inline__ __device__ void SensorProcessor::processDetection(SimulationData& data, SimulationStatistics& statistics, Object* object)
{
    auto enableRelocationScan = object->typeData.cell.signal.channels[Channels::SensorWithRelocationScan] < -NEAR_ZERO;
    if (enableRelocationScan) {
        if (!object->typeData.cell.cellTypeData.sensor.lastMatchAvailable) {
            initialScan(data, statistics, object);
        } else {
            relocateLastMatch(data, statistics, object);
        }
    } else {
        initialScan(data, statistics, object);
    }
}

__inline__ __device__ void SensorProcessor::processTelemetry(SimulationData& data, SimulationStatistics& statistics, Object* object)
{
    if (threadIdx.x == 0) {

        // Create signal if not already existing
        if (object->typeData.cell.signalState != SignalState_Active) {
            SignalProcessor::createEmptySignal(object);
        }

        // Measure cell energy level
        auto cellMinEnergy = ParameterCalculator::calcParameter(cudaSimulationParameters.minCellEnergy, data, object->pos, object->color);
        auto energyAboveMin = max(object->typeData.cell.usableEnergy - cellMinEnergy, 0.0f);
        // Mapping energyAboveMin to [0.0, 1.0]
        // 0    -> 0
        // 10   -> 0.21
        // 50   -> 0.32
        // 100  -> 0.36
        // 1000 -> 0.5
        object->typeData.cell.signal.channels[Channels::SensorTelemetryCellEnergy] = 1.0f - 1.0f / powf(object->typeData.cell.usableEnergy + 1.0f, 0.1f);

        // Measure cell velocity with respect to front angle
        auto refAngle = Math::angleOfVector(SignalProcessor::calcReferenceDirection(data, object));
        auto absFrontAngle = refAngle + object->typeData.cell.frontAngle;
        auto velAngle = Math::angleOfVector(object->vel);
        object->typeData.cell.signal.channels[Channels::SensorTelemetryCellVelAngle] =
            Math::getNormalizedAngle(velAngle - absFrontAngle, -180.0f) / 180.0f;  // Angle: between -1.0 and 1.0

        // Measure cell velocity with
        auto vel = Math::length(object->vel);
        // Mapping velocity to [0.0, 1.0]
        // 0     -> 0
        // 0.001 -> 0.014
        // 0.01  -> 0.12
        // 0.1   -> 0.52
        // 0.5   -> 0.94
        object->typeData.cell.signal.channels[Channels::SensorTelemetryCellVelStrength] = min(log10f(1.0f + vel * 50) / 1.5f, 1.0f);
    }
}

__inline__ __device__ void SensorProcessor::initialScan(SimulationData& data, SimulationStatistics& statistics, Object* object)
{
    __shared__ uint64_t lookupResult;

    __shared__ Object* nearSameCreatureCells[MaxSameNearCreatureCells];
    __shared__ int numNearSameCreatureCells;
    __shared__ float seedAngle;

    if (threadIdx.x == 0) {
        lookupResult = 0xffffffffffffffff;
        seedAngle = data.primaryNumberGen.random(360.0f);

        data.objectMap.getMatchingObjects(
            nearSameCreatureCells, MaxSameNearCreatureCells, numNearSameCreatureCells, object->pos, 4.0f, object->detached, [&](Object* const& otherObject) {
                return otherObject->type == ObjectType_Cell && object->typeData.cell.isSameCreature(&otherObject->typeData.cell);
            });
    }

    __syncthreads();

    auto const& densityMap = data.preprocessedSimulationData.densityMap;

    auto const startRadius = toFloat(object->typeData.cell.cellTypeData.sensor.minRange);
    auto endRadius = min(cudaSimulationParameters.sensorRadius.value[object->color], toFloat(object->typeData.cell.cellTypeData.sensor.maxRange));

    // 1. Near range scan
    if (object->typeData.cell.cellTypeData.sensor.mode == SensorMode_DetectCreature) {

        // Calculate the total number of scan positions in the [-range, range] x [-range, range] region
        auto const nearDistance = toInt(ScanStep);
        int diameter = 2 * nearDistance + 1;
        int totalPositions = diameter * diameter;

        // Each thread scans different positions in parallel
        for (int idx = threadIdx.x; idx < totalPositions; idx += blockDim.x) {
            auto dx = toFloat((idx % diameter) - nearDistance);
            auto dy = toFloat((idx / diameter) - nearDistance);

            auto delta = float2{dx, dy};
            auto angle = Math::angleOfVector(delta);
            float2 scanPos = object->pos + delta;
            data.objectMap.correctPosition(scanPos);

            // Check all cells at this position (including overlapping cells)
            auto distange = Math::length(delta);
            uint64_t matchInfo = getMatchInfo(data, object, scanPos, angle, distange, ScanType::LocateMatch);
            if (matchInfo != 0xffffffffffffffff) {
                if (!isRayBlockedByCreatureConnections(nearSameCreatureCells, numNearSameCreatureCells, object->pos, angle)) {
                    alienAtomicMin64(&lookupResult, matchInfo);
                }
            }
        }
    }
    __syncthreads();

    // 2. Long range scan
    if (lookupResult == 0xffffffffffffffff) {

        // Each thread processes multiple rays if blockDim.x < NumScanRays
        for (int rayIdx = threadIdx.x; rayIdx < NumScanRays; rayIdx += blockDim.x) {

            // First check if ray is blocked by connections of nearby same-creature cells
            auto angle = 360.0f * toFloat(rayIdx) / toFloat(NumScanRays) + seedAngle;
            auto seedDistance = data.primaryNumberGen.random(0, ScanStep);

            if (!isRayBlockedByCreatureConnections(nearSameCreatureCells, numNearSameCreatureCells, object->pos, angle)) {
                for (float distance = seedDistance; distance <= endRadius; distance += ScanStep) {
                    auto delta = Math::unitVectorOfAngle(angle) * distance;
                    auto scanPos = object->pos + delta;
                    data.objectMap.correctPosition(scanPos);

                    if (distance > startRadius) {
                        uint64_t matchInfo = getMatchInfo(data, object, scanPos, angle, distance, ScanType::LocateMatch);
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
        }
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        if (lookupResult != 0xffffffffffffffff) {
            // Create signal if not already existing
            if (object->typeData.cell.signalState != SignalState_Active) {
                SignalProcessor::createEmptySignal(object);
            }

            float distance, absAngle, density;
            uint16_t creatureIdPart;
            unpack(distance, absAngle, density, creatureIdPart, lookupResult);

            auto refAngle = Math::angleOfVector(SignalProcessor::calcReferenceDirection(data, object));
            auto relAngle = Math::getNormalizedAngle(absAngle - refAngle - object->typeData.cell.frontAngle, -180.0f);
            writeSignal(object->typeData.cell.signal, relAngle, density, distance);
            statistics.incNumSensorMatches(object->color);

            auto matchPos = object->pos + Math::unitVectorOfAngle(absAngle) * distance;
            data.objectMap.correctPosition(matchPos);

            // No relocation for structures
            if (object->typeData.cell.cellTypeData.sensor.mode != SensorMode_DetectStructure) {
                object->typeData.cell.cellTypeData.sensor.lastMatchAvailable = true;
                object->typeData.cell.cellTypeData.sensor.lastMatch.creatureId = creatureIdPart;
                object->typeData.cell.cellTypeData.sensor.lastMatch.pos = matchPos;
            }
        } else {
            object->typeData.cell.cellTypeData.sensor.lastMatchAvailable = false;
            object->typeData.cell.signal.channels[Channels::SensorFoundResult] = 0;  // Nothing found
        }
    }
}

__inline__ __device__ void SensorProcessor::relocateLastMatch(SimulationData& data, SimulationStatistics& statistics, Object* object)
{
    // Search grid is (2*RelocationSearchRadius) x (2*RelocationSearchRadius)
    // Each thread handles multiple columns if blockDim.x < (2*RelocationSearchRadius)
    int searchDiameter = 2 * RelocationSearchRadius;

    __shared__ uint64_t lookupResult;
    __shared__ float refAngle;

    if (threadIdx.x == 0) {
        lookupResult = 0xffffffffffffffff;
        refAngle = Math::angleOfVector(SignalProcessor::calcReferenceDirection(data, object));
    }
    __syncthreads();

    auto centerScanPos = object->typeData.cell.cellTypeData.sensor.lastMatch.pos;

    // Each thread handles multiple columns (deltaX values)
    for (int colIdx = threadIdx.x; colIdx < searchDiameter; colIdx += blockDim.x) {
        int deltaX = colIdx - RelocationSearchRadius;

        for (int deltaY = -RelocationSearchRadius; deltaY < RelocationSearchRadius; ++deltaY) {

            auto delta = float2{toFloat(deltaX), toFloat(deltaY)};
            auto scanPos = centerScanPos + delta;
            auto distance = Math::length(delta);
            auto angle = Math::angleOfVector(delta);
            uint64_t matchInfo = getMatchInfo(data, object, scanPos, angle, distance, ScanType::RelocateLastMatch);
            if (matchInfo != 0xffffffffffffffff) {
                alienAtomicMin64(&lookupResult, matchInfo);
                break;
            }
        }
    }
    __syncthreads();

    __shared__ float distance, absAngle, density;
    __shared__ uint16_t creatureIdPart;
    __shared__ float2 direction;
    if (lookupResult != 0xffffffffffffffff) {
        if (threadIdx.x == 0) {
            unpack(distance, absAngle, density, creatureIdPart, lookupResult);
            direction = Math::unitVectorOfAngle(absAngle);
        }
        __syncthreads();

        // Check if ray from sensor to match pos is blocked by structure
        if (distance >= ScanStep) {
            auto const partition = calcSystemThreadPartition(toInt(distance) / ScanStep);
            auto const& densityMap = data.preprocessedSimulationData.densityMap;
            for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
                auto scanDistance = toFloat(index) * ScanStep;
                auto scanPos = data.objectMap.getCorrectedPosition(object->pos + direction * scanDistance);
                if (densityMap.getStructureDensity(scanPos) > 0) {
                    lookupResult = 0xffffffffffffffff;
                    break;
                }
            }
        }

        // Check if match is outside scan range
        if (threadIdx.x == 0) {
            auto startRadius = toFloat(object->typeData.cell.cellTypeData.sensor.minRange);
            auto endRadius = min(cudaSimulationParameters.sensorRadius.value[object->color], toFloat(object->typeData.cell.cellTypeData.sensor.maxRange));
            if (distance < startRadius || distance > endRadius) {
                lookupResult = 0xffffffffffffffff;
            }
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        if (lookupResult != 0xffffffffffffffff) {

            // Create signal if not already existing
            if (object->typeData.cell.signalState != SignalState_Active) {
                SignalProcessor::createEmptySignal(object);
            }

            auto targetPos = object->pos + Math::unitVectorOfAngle(absAngle) * distance;
            auto relAngle = Math::getNormalizedAngle(absAngle - refAngle - object->typeData.cell.frontAngle, -180.0f);
            writeSignal(object->typeData.cell.signal, relAngle, density, distance);

            statistics.incNumSensorMatches(object->color);

            object->typeData.cell.cellTypeData.sensor.lastMatchAvailable = true;
            object->typeData.cell.cellTypeData.sensor.lastMatch.creatureId = creatureIdPart;
            object->typeData.cell.cellTypeData.sensor.lastMatch.pos = targetPos;
        } else {
            object->typeData.cell.cellTypeData.sensor.lastMatchAvailable = false;
            object->typeData.cell.signal.channels[Channels::SensorFoundResult] = 0;  // Nothing found
        }
    }
}

__inline__ __device__ uint64_t
SensorProcessor::getMatchInfo(SimulationData& data, Object* object, float2 const& scanPos, float absAngle, float distance, ScanType scanType)
{
    if (scanType == ScanType::RelocateLastMatch) {
        auto delta = data.objectMap.getCorrectedDirection(scanPos - object->pos);
        distance = Math::length(delta);
        absAngle = Math::angleOfVector(delta);
    }
    absAngle = Math::getNormalizedAngle(absAngle, -180.0f);
    auto const& cell = &object->typeData.cell;

    auto const& mode = cell->cellTypeData.sensor.mode;
    auto const& densityMap = data.preprocessedSimulationData.densityMap;

    if (mode == SensorMode_DetectEnergy) {
        auto const& minDensity = cell->cellTypeData.sensor.modeData.detectEnergy.minDensity;
        auto density = densityMap.getEnergyParticleDensity(scanPos);
        if (density >= minDensity) {
            return pack(distance, absAngle, density);
        }
    } else if (mode == SensorMode_DetectStructure) {
        if (densityMap.getStructureDensity(scanPos) > 0) {
            return pack(distance, absAngle, 1.0f);
        }
    } else if (mode == SensorMode_DetectFreeCell) {
        auto const& minDensity = cell->cellTypeData.sensor.modeData.detectFreeCell.minDensity;
        auto const& restrictToColor = cell->cellTypeData.sensor.modeData.detectFreeCell.restrictToColor;
        auto density = densityMap.getFreeCellDensity(scanPos, restrictToColor);
        if (density >= minDensity) {
            return pack(distance, absAngle, density);
        }
    } else if (mode == SensorMode_DetectCreature) {
        if (scanType == ScanType::LocateMatch) {

            auto const& minNumCells = cell->cellTypeData.sensor.modeData.detectCreature.minNumCells;
            auto const& maxNumCells = cell->cellTypeData.sensor.modeData.detectCreature.maxNumCells;
            auto const& restrictToColor = cell->cellTypeData.sensor.modeData.detectCreature.restrictToColor;
            auto const& restrictToLineage = cell->cellTypeData.sensor.modeData.detectCreature.restrictToLineage;

            auto otherObject = data.objectMap.getFirst(scanPos);
            while (otherObject != nullptr) {
                // Check if this cell is part of a creature (not structure or free object)
                if (otherObject->type == ObjectType_Cell && !cell->isSameCreature(&otherObject->typeData.cell)) {
                    bool matches = true;

                    if (restrictToColor != 255 && otherObject->color != restrictToColor) {
                        matches = false;
                    }
                    if (matches && minNumCells > 0 && otherObject->typeData.cell.creature->numObjects < minNumCells) {
                        matches = false;
                    }
                    if (matches && maxNumCells > 0 && otherObject->typeData.cell.creature->numObjects > maxNumCells) {
                        matches = false;
                    }
                    if (matches && restrictToLineage != LineageRestriction_No) {
                        if (restrictToLineage == LineageRestriction_SameLineage) {
                            if (object->typeData.cell.creature->genome->lineageId != otherObject->typeData.cell.creature->genome->lineageId) {
                                matches = false;
                            }
                        } else if (restrictToLineage == LineageRestriction_OtherLineage) {
                            if (object->typeData.cell.creature->genome->lineageId == otherObject->typeData.cell.creature->genome->lineageId) {
                                matches = false;
                            }
                        }
                    }

                    if (matches) {
                        uint16_t creatureIdPart = static_cast<uint16_t>(otherObject->typeData.cell.creature->id & 0xFFFF);
                        float density = calcCreatureDensityFromNumCells(otherObject->typeData.cell.creature->numObjects);
                        return pack(distance, absAngle, density, creatureIdPart);
                    }
                }
                otherObject = otherObject->nextObject;
            }

            // Else: ScanType::Relocation
        } else {
            auto& sensor = cell->cellTypeData.sensor;
            auto otherObject = data.objectMap.getFirst(scanPos);
            while (otherObject != nullptr) {
                if (otherObject->type == ObjectType_Cell && (otherObject->typeData.cell.creature->id & 0xffff) == sensor.lastMatch.creatureId) {
                    uint16_t creatureIdPart = static_cast<uint16_t>(otherObject->typeData.cell.creature->id & 0xffff);
                    float density = calcCreatureDensityFromNumCells(otherObject->typeData.cell.creature->numObjects);
                    return pack(distance, absAngle, density, creatureIdPart);
                }
                otherObject = otherObject->nextObject;
            }
        }
    }
    return 0xffffffffffffffff;
}

__inline__ __device__ bool
SensorProcessor::isRayBlockedByCreatureConnections(Object** nearSameCreatureCells, int numNearSameCreatureCells, float2 const& rayOrigin, float angle)
{
    auto rayEnd = rayOrigin + Math::unitVectorOfAngle(angle) * RayBlockingTestLength;
    for (int i = 0; i < numNearSameCreatureCells; ++i) {
        auto nearObject = nearSameCreatureCells[i];
        for (int j = 0, k = nearObject->numConnections; j < k; ++j) {
            auto& connectedNearObject = nearObject->connections[j].object;
            if (Math::crossing(nearObject->pos, connectedNearObject->pos, rayOrigin, rayEnd)) {
                return true;
            }
        }
    }
    return false;
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
    signal.channels[Channels::SensorFoundResult] = 1;                              // Something found
    signal.channels[Channels::SensorAngle] = angle / 180.0f;                       // Angle: between -1.0 and 1.0
    signal.channels[Channels::SensorMass] = min(1.0f, density);                    // Normalized density (1.0 = 64 cells in 8x8 region)
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

// Converts creature cell count to density value for SensorMode_DetectCreature
// Non-linear scale: 0.5 means 30 cells, 0.75 means 60 cells, 1.0 means 120 cells and so on.
// Formula: density = 0.25 * log2(numCells / 30) + 0.5, clamped to [0.0, 1.0]
__inline__ __device__ float SensorProcessor::calcCreatureDensityFromNumCells(uint32_t numCells)
{
    if (numCells == 0) {
        return 0.0f;
    }
    numCells = max(8, numCells);  // Below 8 cells formular would yield negative density
    float density = 0.25f * log2f(static_cast<float>(numCells) / 30.0f) + 0.5f;

    return min(1.0f, max(0.0f, density));
}
