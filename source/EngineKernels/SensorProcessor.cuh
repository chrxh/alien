#pragma once

#include "SimulationData.cuh"

class SensorProcessor
{
public:
    __inline__ __device__ static void process(SimulationData& data, SimulationStatistics& statistics);

private:
    static int constexpr MaxSameNearCreatureCells = 9 * 9;

    struct ScanState
    {
        uint64_t lookupResult;
        Object* nearSameCreatureCells[MaxSameNearCreatureCells];
        int numNearSameCreatureCells;
        float seedAngle;
    };

    __inline__ __device__ static void processCell(SimulationData& data, SimulationStatistics& statistics, Object* object);
    __inline__ __device__ static void processDetection(SimulationData& data, SimulationStatistics& statistics, Object* object);

    __inline__ __device__ static void initialScan(SimulationData& data, SimulationStatistics& statistics, Object* object);
    __inline__ __device__ static void scanSolid(SimulationData& data, Object* object, ScanState& state, float startRadius, float endRadius);
    __inline__ __device__ static void scanEnergy(SimulationData& data, Object* object, ScanState& state, float startRadius, float endRadius);
    __inline__ __device__ static void scanFreeCell(SimulationData& data, Object* object, ScanState& state, float startRadius, float endRadius);
    __inline__ __device__ static void scanCreature(SimulationData& data, Object* object, ScanState& state, float startRadius, float endRadius);
    __inline__ __device__ static void scanCreatureNearRange(SimulationData& data, Object* object, ScanState& state, float startRadius, float endRadius);
    template <typename MatchFunc>
    __inline__ __device__ static void
    scanRays(SimulationData& data, Object* object, ScanState& state, float startRadius, float endRadius, MatchFunc const& getMatchInfo);

    __inline__ __device__ static void relocateLastMatch(SimulationData& data, SimulationStatistics& statistics, Object* object);
    __inline__ __device__ static uint64_t getRelocationMatchInfo(SimulationData& data, Object* object, float2 const& scanPos);

    __inline__ __device__ static void publishMatch(SimulationData& data, Object* object, uint64_t lookupResult);
    __inline__ __device__ static void publishNoMatch(Object* object);

    __inline__ __device__ static void registerDetections(SimulationData& data, Object* object);
    template <typename Filter>
    __inline__ __device__ static Creature* findNearestCreature(SimulationData& data, Object* object, float2 const& pos, Filter const& filter);
    __inline__ __device__ static void registerDetection(SimulationData& data, Creature* creature, uint16_t creatureIdPart, uint16_t restrictToColors);

    __inline__ __device__ static uint64_t matchEnergy(SimulationData& data, float minDensity, float2 const& scanPos, float2 const& delta, float distance);
    __inline__ __device__ static uint64_t
    matchFreeCell(SimulationData& data, float minDensity, uint16_t restrictToColors, float2 const& scanPos, float2 const& delta, float distance);
    __inline__ __device__ static uint64_t matchCreature(SimulationData& data, Object* object, float2 const& scanPos, float2 const& delta, float distance);
    __inline__ __device__ static uint64_t
    matchLastMatchedCreature(SimulationData& data, Object* object, float2 const& scanPos, float2 const& delta, float distance);
    __inline__ __device__ static bool isMatchingCreature(Object* object, Object* otherObject);

    __inline__ __device__ static float calcRayAngle(int rayIdx, float seedAngle);
    __inline__ __device__ static float calcAbsAngle(float2 const& delta);
    __inline__ __device__ static bool isRayBlockedByCreatureConnections(ScanState const& state, float2 const& rayOrigin, float angle);

    __inline__ __device__ static bool isRayBlockedBySolid(SimulationData& data, float2 const& rayOrigin, float angle, float distance);
    __inline__ __device__ static float calcFreeRayLength(SimulationData& data, float2 const& rayOrigin, float angle, float maxLength);
    __inline__ __device__ static float2 calcRayRangeInSquare(float2 const& pos, float2 const& direction, int squareSize);
    __inline__ __device__ static bool isSolidNearSegment(SimulationData& data, float2 const& pos, float2 const& direction, float2 const& range);
    __inline__ __device__ static bool isSolidNearPosition(SimulationData& data, float2 const& pos, float2 const& direction, float distance);
    __inline__ __device__ static float
    findSolidInSegment(SimulationData& data, float2 const& origin, float2 const& direction, float startDistance, float endDistance, int numRefinements);
    __inline__ __device__ static float findSolidAlongRay(
        SimulationData& data,
        float2 const& origin,
        float angle,
        float startRadius,
        float endRadius,
        float seedDistance,
        int numRefinements,
        uint64_t const* bestMatch = nullptr);
    __inline__ __device__ static bool isBeyondBestMatch(uint64_t const* bestMatch, float distance);

    __inline__ __device__ static uint64_t pack(float distance, float angle, float density, uint16_t misc = 0);
    __inline__ __device__ static void unpack(float& distance, float& angle, float& density, uint16_t& misc, uint64_t bytes);

    __inline__ __device__ static void writeSignal(NeuralActivity& neuralActivity, float angle, float density, float distance);

    __inline__ __device__ static uint16_t convertAngleToUint16(float angle);
    __inline__ __device__ static float convertUint16ToAngle(uint16_t b);

    __inline__ __device__ static float calcCreatureDensityFromNumCells(uint32_t numCells);

    static uint64_t constexpr NoMatch = 0xffffffffffffffff;
    static int constexpr NumScanRays = 64;
    static int constexpr RelocationSearchRadius = 32;  // Search grid is (2*radius) x (2*radius) = 64x64
    static float constexpr ScanStep = 8.0f;
    static float constexpr RayBlockingTestLength = 10.0f;

    static float constexpr SolidScanStep = 1.0f;
    static float constexpr SolidHitRadius = 0.75f;
    static int constexpr NumSolidRefinements = 6;
    static int constexpr NumOcclusionRefinements = 2;  // Occlusion needs a lower precision than the distance of a detected solid
    static float constexpr SquareTransitionEpsilon = 0.05f;

    static float constexpr DistanceScale = 64.0f;  // Fixed-point resolution of the packed distance, covers distances up to 1023

    static int constexpr MaxNearbyCreatures = 3;
    static float constexpr NearbyCreatureRadius = 4.0f;
    static uint32_t constexpr NoNearbyCreature = 0xffffffff;
    static uint32_t constexpr MinDetectedByCapacity = 4;
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
        isTriggered = NeuronProcessor::isAutoOrManuallyTriggered(data, object, object->typeData.cell.cellTypeData.sensor.autoTrigger);
        if (object->typeData.cell.frontAngle == VALUE_NOT_SET_FLOAT) {
            isTriggered = false;
        }
    }
    __syncthreads();

    if (isTriggered) {
        processDetection(data, statistics, object);
    }
}

__inline__ __device__ void SensorProcessor::processDetection(SimulationData& data, SimulationStatistics& statistics, Object* object)
{
    auto const& sensor = object->typeData.cell.cellTypeData.sensor;
    auto enableRelocationScan = object->typeData.cell.neuralActivity.signals[Channels::SensorWithRelocationScan] < -NEAR_ZERO;
    if (enableRelocationScan && sensor.lastMatchAvailable) {
        relocateLastMatch(data, statistics, object);
    } else {
        initialScan(data, statistics, object);
    }
    __syncthreads();

    if (sensor.tagForAttackers && sensor.mode == SensorMode_DetectCreature && sensor.lastMatchAvailable) {
        registerDetections(data, object);
    }
}

__inline__ __device__ void SensorProcessor::initialScan(SimulationData& data, SimulationStatistics& statistics, Object* object)
{
    __shared__ ScanState state;

    if (threadIdx.x == 0) {
        state.lookupResult = NoMatch;
        state.seedAngle = data.primaryNumberGen.random(360.0f);
        state.numNearSameCreatureCells = 0;
    }
    __syncthreads();

    data.objectGrid.executeForEach_block(object->pos, 4.0f, object->detached(), [&](Object* const& otherObject) {
        if (otherObject->type == ObjectType_Cell && object->typeData.cell.isSameCreature(&otherObject->typeData.cell)) {
            auto index = atomicAdd(&state.numNearSameCreatureCells, 1);
            if (index < MaxSameNearCreatureCells) {
                state.nearSameCreatureCells[index] = otherObject;
            }
        }
    });
    __syncthreads();

    if (threadIdx.x == 0) {
        state.numNearSameCreatureCells = min(state.numNearSameCreatureCells, MaxSameNearCreatureCells);
    }
    __syncthreads();

    auto const& sensor = object->typeData.cell.cellTypeData.sensor;
    auto startRadius = toFloat(sensor.minRange);
    auto endRadius = min(cudaSimulationParameters.sensorRadius.value[object->color], toFloat(sensor.maxRange));

    switch (sensor.mode) {
    case SensorMode_DetectSolid:
        scanSolid(data, object, state, startRadius, endRadius);
        break;
    case SensorMode_DetectEnergy:
        scanEnergy(data, object, state, startRadius, endRadius);
        break;
    case SensorMode_DetectFreeCell:
        scanFreeCell(data, object, state, startRadius, endRadius);
        break;
    case SensorMode_DetectCreature:
        scanCreature(data, object, state, startRadius, endRadius);
        break;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        if (state.lookupResult != NoMatch) {
            publishMatch(data, object, state.lookupResult);
        } else {
            publishNoMatch(object);
        }
    }
}

__inline__ __device__ void SensorProcessor::scanSolid(SimulationData& data, Object* object, ScanState& state, float startRadius, float endRadius)
{
    for (int rayIdx = threadIdx.x; rayIdx < NumScanRays; rayIdx += blockDim.x) {
        auto angle = calcRayAngle(rayIdx, state.seedAngle);
        if (isRayBlockedByCreatureConnections(state, object->pos, angle)) {
            continue;
        }
        auto solidDistance = findSolidAlongRay(
            data, object->pos, angle, startRadius, endRadius, data.primaryNumberGen.random(0, SolidScanStep), NumSolidRefinements, &state.lookupResult);
        if (solidDistance >= 0) {
            alienAtomicMin64(&state.lookupResult, pack(solidDistance, Math::getNormalizedAngle(angle, -180.0f), 1.0f));
        }
    }
}

__inline__ __device__ void SensorProcessor::scanEnergy(SimulationData& data, Object* object, ScanState& state, float startRadius, float endRadius)
{
    auto minDensity = object->typeData.cell.cellTypeData.sensor.modeData.detectEnergy.minDensity;
    scanRays(data, object, state, startRadius, endRadius, [&](float2 const& scanPos, float2 const& delta, float distance) {
        return matchEnergy(data, minDensity, scanPos, delta, distance);
    });
}

__inline__ __device__ void SensorProcessor::scanFreeCell(SimulationData& data, Object* object, ScanState& state, float startRadius, float endRadius)
{
    auto const& mode = object->typeData.cell.cellTypeData.sensor.modeData.detectFreeCell;
    scanRays(data, object, state, startRadius, endRadius, [&](float2 const& scanPos, float2 const& delta, float distance) {
        return matchFreeCell(data, mode.minDensity, mode.restrictToColors, scanPos, delta, distance);
    });
}

__inline__ __device__ void SensorProcessor::scanCreature(SimulationData& data, Object* object, ScanState& state, float startRadius, float endRadius)
{
    scanCreatureNearRange(data, object, state, startRadius, endRadius);
    __syncthreads();

    if (state.lookupResult == NoMatch) {
        scanRays(data, object, state, startRadius, endRadius, [&](float2 const& scanPos, float2 const& delta, float distance) {
            return matchCreature(data, object, scanPos, delta, distance);
        });
    }
}

__inline__ __device__ void SensorProcessor::scanCreatureNearRange(SimulationData& data, Object* object, ScanState& state, float startRadius, float endRadius)
{
    auto nearDistance = toInt(ScanStep);
    int diameter = 2 * nearDistance + 1;
    int totalPositions = diameter * diameter;

    // Each thread scans different positions in parallel
    for (int idx = threadIdx.x; idx < totalPositions; idx += blockDim.x) {
        auto dx = toFloat((idx % diameter) - nearDistance);
        auto dy = toFloat((idx / diameter) - nearDistance);

        auto delta = float2{dx, dy};
        auto distance = Math::length(delta);
        if (distance <= startRadius || distance > endRadius) {
            continue;
        }
        float2 scanPos = object->pos + delta;
        data.world.correctPosition(scanPos);

        // Check all cells at this position (including overlapping cells)
        auto matchInfo = matchCreature(data, object, scanPos, delta, distance);
        if (matchInfo != NoMatch) {
            auto angle = Math::angleOfVector(delta);
            if (!isRayBlockedByCreatureConnections(state, object->pos, angle) && !isRayBlockedBySolid(data, object->pos, angle, distance)) {
                alienAtomicMin64(&state.lookupResult, matchInfo);
            }
        }
    }
}

template <typename MatchFunc>
__inline__ __device__ void
SensorProcessor::scanRays(SimulationData& data, Object* object, ScanState& state, float startRadius, float endRadius, MatchFunc const& getMatchInfo)
{
    // Each thread processes multiple rays if blockDim.x < NumScanRays
    for (int rayIdx = threadIdx.x; rayIdx < NumScanRays; rayIdx += blockDim.x) {
        auto angle = calcRayAngle(rayIdx, state.seedAngle);
        if (isRayBlockedByCreatureConnections(state, object->pos, angle)) {
            continue;
        }

        auto freeLength = endRadius;
        auto direction = Math::unitVectorOfAngle(angle);

        for (float distance = data.primaryNumberGen.random(0, ScanStep); distance <= freeLength && !isBeyondBestMatch(&state.lookupResult, distance);
             distance += ScanStep) {
            auto delta = direction * distance;
            auto scanPos = data.world.getCorrectedPosition(object->pos + delta);

            if (distance > startRadius) {
                auto matchInfo = getMatchInfo(scanPos, delta, distance);
                if (matchInfo != NoMatch) {
                    if (!isRayBlockedBySolid(data, object->pos, angle, distance)) {
                        alienAtomicMin64(&state.lookupResult, matchInfo);
                    }
                    break;
                }
            }

            auto range = calcRayRangeInSquare(scanPos, direction, OccupancyGrid::TileSize);
            if (isSolidNearSegment(data, scanPos, direction, range)) {
                auto solidDistance = findSolidInSegment(
                    data,
                    object->pos,
                    direction,
                    max(distance + range.x - SolidHitRadius, 0.0f),
                    min(distance + range.y + SolidHitRadius, endRadius),
                    NumOcclusionRefinements);
                if (solidDistance >= 0) {
                    freeLength = min(freeLength, solidDistance);
                }
            }
        }
    }
}

__inline__ __device__ void SensorProcessor::relocateLastMatch(SimulationData& data, SimulationStatistics& statistics, Object* object)
{
    // Search grid is (2*RelocationSearchRadius) x (2*RelocationSearchRadius)
    // Each thread handles multiple columns if blockDim.x < (2*RelocationSearchRadius)
    int searchDiameter = 2 * RelocationSearchRadius;

    __shared__ uint64_t lookupResult;
    if (threadIdx.x == 0) {
        lookupResult = NoMatch;
    }
    __syncthreads();

    auto centerScanPos = object->typeData.cell.cellTypeData.sensor.lastMatch.pos;

    // Each thread handles multiple columns (deltaX values)
    for (int colIdx = threadIdx.x; colIdx < searchDiameter; colIdx += blockDim.x) {
        int deltaX = colIdx - RelocationSearchRadius;

        for (int deltaY = -RelocationSearchRadius; deltaY < RelocationSearchRadius; ++deltaY) {
            auto scanPos = centerScanPos + float2{toFloat(deltaX), toFloat(deltaY)};
            auto matchInfo = getRelocationMatchInfo(data, object, scanPos);
            if (matchInfo != NoMatch) {
                alienAtomicMin64(&lookupResult, matchInfo);
                break;
            }
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        auto isValid = false;
        if (lookupResult != NoMatch) {
            float distance, absAngle, density;
            uint16_t creatureIdPart;
            unpack(distance, absAngle, density, creatureIdPart, lookupResult);

            auto startRadius = toFloat(object->typeData.cell.cellTypeData.sensor.minRange);
            auto endRadius = min(cudaSimulationParameters.sensorRadius.value[object->color], toFloat(object->typeData.cell.cellTypeData.sensor.maxRange));
            isValid = distance >= startRadius && distance <= endRadius && !isRayBlockedBySolid(data, object->pos, absAngle, distance);
        }
        if (isValid) {
            publishMatch(data, object, lookupResult);
        } else {
            publishNoMatch(object);
        }
    }
}

__inline__ __device__ uint64_t SensorProcessor::getRelocationMatchInfo(SimulationData& data, Object* object, float2 const& scanPos)
{
    auto const& sensor = object->typeData.cell.cellTypeData.sensor;
    auto delta = data.world.getCorrectedDirection(scanPos - object->pos);
    auto distance = Math::length(delta);

    switch (sensor.mode) {
    case SensorMode_DetectEnergy:
        return matchEnergy(data, sensor.modeData.detectEnergy.minDensity, scanPos, delta, distance);
    case SensorMode_DetectFreeCell:
        return matchFreeCell(data, sensor.modeData.detectFreeCell.minDensity, sensor.modeData.detectFreeCell.restrictToColors, scanPos, delta, distance);
    case SensorMode_DetectCreature:
        return matchLastMatchedCreature(data, object, scanPos, delta, distance);
    }
    return NoMatch;
}

__inline__ __device__ void SensorProcessor::publishMatch(SimulationData& data, Object* object, uint64_t lookupResult)
{
    float distance, absAngle, density;
    uint16_t creatureIdPart;
    unpack(distance, absAngle, density, creatureIdPart, lookupResult);

    auto& cell = object->typeData.cell;
    auto refAngle = Math::angleOfVector(ObjectConnectionProcessor::calcReferenceDirection(data, object));
    auto relAngle = Math::getNormalizedAngle(absAngle - refAngle - cell.frontAngle, -180.0f);
    writeSignal(cell.neuralActivity, relAngle, density, distance);

    // No relocation for solids
    if (cell.cellTypeData.sensor.mode != SensorMode_DetectSolid) {
        auto matchPos = object->pos + Math::unitVectorOfAngle(absAngle) * distance;
        data.world.correctPosition(matchPos);

        cell.cellTypeData.sensor.lastMatchAvailable = true;
        cell.cellTypeData.sensor.lastMatch.creatureIdPart = creatureIdPart;
        cell.cellTypeData.sensor.lastMatch.pos = matchPos;
    }
}

__inline__ __device__ void SensorProcessor::publishNoMatch(Object* object)
{
    object->typeData.cell.cellTypeData.sensor.lastMatchAvailable = false;
    object->typeData.cell.neuralActivity.signals[Channels::SensorFoundResult] = 0;  // Nothing found
}

// Registers the sensor's creature at the matched creature and at the matching creatures nearest to the match, a creature counts with its nearest cell
__inline__ __device__ void SensorProcessor::registerDetections(SimulationData& data, Object* object)
{
    __shared__ Creature* detectedCreatures[1 + MaxNearbyCreatures];

    auto& cell = object->typeData.cell;
    auto const& sensor = cell.cellTypeData.sensor;
    auto creatureIdPart = static_cast<uint16_t>(cell.creature->id & 0xffff);
    auto restrictToColors = sensor.modeData.detectCreature.restrictToColors;

    if (threadIdx.x == 0) {
        for (auto& detectedCreature : detectedCreatures) {
            detectedCreature = nullptr;
        }
    }
    __syncthreads();

    auto matchedCreature = findNearestCreature(data, object, sensor.lastMatch.pos, [&](Object* otherObject) {
        return otherObject->type == ObjectType_Cell && !cell.isSameCreature(&otherObject->typeData.cell)
            && (otherObject->typeData.cell.creature->id & 0xffff) == sensor.lastMatch.creatureIdPart;
    });
    if (threadIdx.x == 0 && matchedCreature != nullptr) {
        detectedCreatures[0] = matchedCreature;
        registerDetection(data, matchedCreature, creatureIdPart, restrictToColors);
    }
    __syncthreads();

    for (int i = 1; i <= MaxNearbyCreatures; ++i) {
        auto nearbyCreature = findNearestCreature(data, object, sensor.lastMatch.pos, [&](Object* otherObject) {
            if (!isMatchingCreature(object, otherObject)) {
                return false;
            }
            for (auto const& detectedCreature : detectedCreatures) {
                if (detectedCreature == otherObject->typeData.cell.creature) {
                    return false;
                }
            }
            return true;
        });
        if (nearbyCreature == nullptr) {
            return;
        }
        if (threadIdx.x == 0) {
            detectedCreatures[i] = nearbyCreature;
            registerDetection(data, nearbyCreature, creatureIdPart, restrictToColors);
        }
        __syncthreads();
    }
}

template <typename Filter>
__inline__ __device__ Creature* SensorProcessor::findNearestCreature(SimulationData& data, Object* object, float2 const& pos, Filter const& filter)
{
    __shared__ uint32_t nearestCreatureKey;
    __shared__ Creature* nearestCreature;

    if (threadIdx.x == 0) {
        nearestCreatureKey = NoNearbyCreature;
        nearestCreature = nullptr;
    }
    __syncthreads();

    auto creatureKey = NoNearbyCreature;
    Creature* creature = nullptr;
    data.objectGrid.executeForEach_block(pos, NearbyCreatureRadius, object->detached(), [&](Object* const& otherObject) {
        if (!filter(otherObject)) {
            return;
        }
        auto otherCreature = otherObject->typeData.cell.creature;
        auto distance = Math::length(data.world.getCorrectedDirection(otherObject->pos - pos));
        auto key = static_cast<uint32_t>(distance * DistanceScale) << 16 | static_cast<uint16_t>(otherCreature->id & 0xffff);
        if (key < creatureKey) {
            creatureKey = key;
            creature = otherCreature;
        }
    });
    atomicMin(&nearestCreatureKey, creatureKey);
    __syncthreads();

    if (creature != nullptr && creatureKey == nearestCreatureKey) {
        nearestCreature = creature;
    }
    __syncthreads();

    auto result = nearestCreature;
    __syncthreads();
    return result;
}

__inline__ __device__ void SensorProcessor::registerDetection(SimulationData& data, Creature* creature, uint16_t creatureIdPart, uint16_t restrictToColors)
{
    creature->getLock();

    auto numDetections = creature->numDetectedBy;
    auto index = 0u;
    while (index < numDetections && creature->detectedBy[index].creatureIdPart != creatureIdPart) {
        ++index;
    }
    if (index < numDetections) {
        creature->detectedBy[index].restrictToColors |= restrictToColors;
    } else {
        if (numDetections == creature->detectedByCapacity) {
            auto newCapacity = max(MinDetectedByCapacity, 2 * numDetections);
            auto newDetectedBy = data.entities.heap.getTypedSubArray<SensorDetection>(newCapacity);
            for (uint32_t i = 0; i < numDetections; ++i) {
                newDetectedBy[i] = creature->detectedBy[i];
            }
            creature->detectedBy = newDetectedBy;
            creature->detectedByCapacity = newCapacity;
        }
        creature->detectedBy[numDetections] = SensorDetection{creatureIdPart, restrictToColors};
        creature->numDetectedBy = numDetections + 1;
    }

    creature->releaseLock();
}

__inline__ __device__ uint64_t SensorProcessor::matchEnergy(SimulationData& data, float minDensity, float2 const& scanPos, float2 const& delta, float distance)
{
    auto density = data.preprocessedSimulationData.densityGrid.getEnergyParticleDensity(scanPos);
    if (density >= minDensity) {
        return pack(distance, calcAbsAngle(delta), density);
    }
    return NoMatch;
}

__inline__ __device__ uint64_t
SensorProcessor::matchFreeCell(SimulationData& data, float minDensity, uint16_t restrictToColors, float2 const& scanPos, float2 const& delta, float distance)
{
    auto density = data.preprocessedSimulationData.densityGrid.getFreeCellDensity(scanPos, restrictToColors);
    if (density >= minDensity) {
        return pack(distance, calcAbsAngle(delta), density);
    }
    return NoMatch;
}

__inline__ __device__ uint64_t SensorProcessor::matchCreature(SimulationData& data, Object* object, float2 const& scanPos, float2 const& delta, float distance)
{
    auto records = data.objectGrid.getRecords();
    int otherIndex = data.objectGrid.getFirstIndex(scanPos);
    while (otherIndex >= 0) {
        auto const& otherRecord = records[otherIndex];
        auto otherObject = otherRecord.self;
        if (isMatchingCreature(object, otherObject)) {
            uint16_t creatureIdPart = static_cast<uint16_t>(otherObject->typeData.cell.creature->id & 0xFFFF);
            float density = calcCreatureDensityFromNumCells(otherObject->typeData.cell.creature->numCells);
            return pack(distance, calcAbsAngle(delta), density, creatureIdPart);
        }
        otherIndex = otherRecord.nextObjectIndex;
    }
    return NoMatch;
}

__inline__ __device__ uint64_t
SensorProcessor::matchLastMatchedCreature(SimulationData& data, Object* object, float2 const& scanPos, float2 const& delta, float distance)
{
    auto const& sensor = object->typeData.cell.cellTypeData.sensor;
    auto records = data.objectGrid.getRecords();
    int otherIndex = data.objectGrid.getFirstIndex(scanPos);
    while (otherIndex >= 0) {
        auto const& otherRecord = records[otherIndex];
        auto otherObject = otherRecord.self;
        if (otherObject->type == ObjectType_Cell && (otherObject->typeData.cell.creature->id & 0xffff) == sensor.lastMatch.creatureIdPart) {
            uint16_t creatureIdPart = static_cast<uint16_t>(otherObject->typeData.cell.creature->id & 0xffff);
            float density = calcCreatureDensityFromNumCells(otherObject->typeData.cell.creature->numCells);
            return pack(distance, calcAbsAngle(delta), density, creatureIdPart);
        }
        otherIndex = otherRecord.nextObjectIndex;
    }
    return NoMatch;
}

__inline__ __device__ bool SensorProcessor::isMatchingCreature(Object* object, Object* otherObject)
{
    if (otherObject->type != ObjectType_Cell) {
        return false;
    }
    auto& cell = object->typeData.cell;
    auto& otherCell = otherObject->typeData.cell;
    if (cell.isSameCreature(&otherCell)) {
        return false;
    }
    auto const& detectCreature = cell.cellTypeData.sensor.modeData.detectCreature;
    if (!((detectCreature.restrictToColors >> otherObject->color) & 1)) {
        return false;
    }
    if (detectCreature.minNumCells > 0 && otherCell.creature->numCells < detectCreature.minNumCells) {
        return false;
    }
    if (detectCreature.maxNumCells > 0 && otherCell.creature->numCells > detectCreature.maxNumCells) {
        return false;
    }
    if (detectCreature.restrictToLineage == LineageRestriction_RelatedLineage && !cell.creature->isSameLineage(otherCell.creature)) {
        return false;
    }
    if (detectCreature.restrictToLineage == LineageRestriction_UnrelatedLineage && cell.creature->isSameLineage(otherCell.creature)) {
        return false;
    }
    return true;
}

__inline__ __device__ float SensorProcessor::calcRayAngle(int rayIdx, float seedAngle)
{
    return 360.0f * toFloat(rayIdx) / toFloat(NumScanRays) + seedAngle;
}

// Calculated only when needed because of the expensive asinf
__inline__ __device__ float SensorProcessor::calcAbsAngle(float2 const& delta)
{
    return Math::getNormalizedAngle(Math::angleOfVector(delta), -180.0f);
}

__inline__ __device__ bool SensorProcessor::isRayBlockedByCreatureConnections(ScanState const& state, float2 const& rayOrigin, float angle)
{
    auto direction = Math::unitVectorOfAngle(angle);
    auto rayEnd = rayOrigin + direction * RayBlockingTestLength;
    auto calcSideOfRay = [&](float2 const& pos) { return direction.x * (pos.y - rayOrigin.y) - direction.y * (pos.x - rayOrigin.x); };

    for (int i = 0; i < state.numNearSameCreatureCells; ++i) {
        auto nearObject = state.nearSameCreatureCells[i];
        auto sideOfNearObject = calcSideOfRay(nearObject->pos);
        for (int j = 0, k = nearObject->numConnections; j < k; ++j) {
            auto& connectedNearObject = nearObject->connections[j].object;

            // A connection with both ends on the same side of the ray cannot cross it
            if (sideOfNearObject * calcSideOfRay(connectedNearObject->pos) > 0) {
                continue;
            }
            if (Math::crossing(nearObject->pos, connectedNearObject->pos, rayOrigin, rayEnd)) {
                return true;
            }
        }
    }
    return false;
}

__inline__ __device__ bool SensorProcessor::isRayBlockedBySolid(SimulationData& data, float2 const& rayOrigin, float angle, float distance)
{
    return calcFreeRayLength(data, rayOrigin, angle, distance) < distance;
}

__inline__ __device__ float SensorProcessor::calcFreeRayLength(SimulationData& data, float2 const& rayOrigin, float angle, float maxLength)
{
    auto solidDistance = findSolidAlongRay(data, rayOrigin, angle, -1.0f, maxLength, 0.0f, NumOcclusionRefinements);
    return solidDistance >= 0 ? min(maxLength, solidDistance) : maxLength;
}

// Distances relative to pos at which the ray enters and leaves the square containing pos, the squares are aligned to multiples of their size
__inline__ __device__ float2 SensorProcessor::calcRayRangeInSquare(float2 const& pos, float2 const& direction, int squareSize)
{
    auto size = toFloat(squareSize);

    auto enter = -2.0f * size;
    auto leave = 2.0f * size;
    auto clip = [&](float position, float directionComponent) {
        if (fabsf(directionComponent) > NEAR_ZERO) {
            auto squareStart = floorf(position / size) * size;
            auto invDirectionComponent = 1.0f / directionComponent;
            auto t1 = (squareStart - position) * invDirectionComponent;
            auto t2 = (squareStart + size - position) * invDirectionComponent;
            enter = max(enter, min(t1, t2));
            leave = min(leave, max(t1, t2));
        }
    };
    clip(pos.x, direction.x);
    clip(pos.y, direction.y);
    return {enter, leave};
}

// Checks the positions around the part of the ray between the distances range.x and range.y relative to pos, the part must lie in the tile of pos
__inline__ __device__ bool SensorProcessor::isSolidNearSegment(SimulationData& data, float2 const& pos, float2 const& direction, float2 const& range)
{
    if (!data.solidGrid.hasSolidNearBlockOf(pos)) {
        return false;
    }
    auto start = pos + direction * range.x;
    auto end = pos + direction * range.y;
    int2 minPos{floorInt(min(start.x, end.x) - SolidHitRadius), floorInt(min(start.y, end.y) - SolidHitRadius)};
    int2 maxPos{floorInt(max(start.x, end.x) + SolidHitRadius), floorInt(max(start.y, end.y) + SolidHitRadius)};
    return data.solidGrid.hasSolid(minPos, maxPos);
}

// Solids behind the ray origin are ignored, otherwise a solid touching the sensor would block every ray
__inline__ __device__ bool SensorProcessor::isSolidNearPosition(SimulationData& data, float2 const& pos, float2 const& direction, float distance)
{
    auto records = data.objectGrid.getRecords();
    int2 const minCell{floorInt(pos.x - SolidHitRadius), floorInt(pos.y - SolidHitRadius)};
    int2 const maxCell{floorInt(pos.x + SolidHitRadius), floorInt(pos.y + SolidHitRadius)};
    if (!data.solidGrid.hasSolid(minCell, maxCell)) {
        return false;
    }
    for (int cellY = minCell.y; cellY <= maxCell.y; ++cellY) {
        for (int cellX = minCell.x; cellX <= maxCell.x; ++cellX) {
            int2 cell{cellX, cellY};
            data.world.correctPosition(cell);
            if (!data.solidGrid.hasSolid(cell)) {
                continue;
            }
            auto index = data.objectGrid.getFirstIndex(cell);
            for (int level = 0; level < 10 && index >= 0; ++level) {
                auto const& record = records[index];
                if (record.type == ObjectType_Solid) {
                    auto delta = record.self->pos - pos;
                    data.world.correctDirection(delta);
                    if (delta.x * delta.x + delta.y * delta.y <= SolidHitRadius * SolidHitRadius && Math::dot(delta, direction) + distance >= 0.0f) {
                        return true;
                    }
                }
                index = record.nextObjectIndex;
            }
        }
    }
    return false;
}

// Returns the distance at which the ray enters the neighborhood of a solid (found by bisection to a fraction of a unit) or a negative value if there is none
__inline__ __device__ float SensorProcessor::findSolidInSegment(
    SimulationData& data,
    float2 const& origin,
    float2 const& direction,
    float startDistance,
    float endDistance,
    int numRefinements)
{
    auto isHit = [&](float distance) { return isSolidNearPosition(data, data.world.getCorrectedPosition(origin + direction * distance), direction, distance); };

    for (float distance = startDistance; distance <= endDistance; distance += SolidScanStep) {
        if (!isHit(distance)) {
            continue;
        }

        auto lower = max(distance - SolidScanStep, 0.0f);
        auto upper = distance;
        if (isHit(lower)) {
            return lower;
        }
        for (int i = 0; i < numRefinements; ++i) {
            auto middle = 0.5f * (lower + upper);
            if (isHit(middle)) {
                upper = middle;
            } else {
                lower = middle;
            }
        }
        return upper;
    }
    return -1.0f;
}

// The ray crosses blocks without solids in one step and is followed tile by tile through the other blocks.
// Only tiles with solids in reach of the ray are examined in detail.
__inline__ __device__ float SensorProcessor::findSolidAlongRay(
    SimulationData& data,
    float2 const& origin,
    float angle,
    float startRadius,
    float endRadius,
    float seedDistance,
    int numRefinements,
    uint64_t const* bestMatch)
{
    auto direction = Math::unitVectorOfAngle(angle);

    auto distance = seedDistance;
    while (distance <= endRadius) {
        auto scanPos = data.world.getCorrectedPosition(origin + direction * distance);
        if (!data.solidGrid.hasSolidNearBlockOf(scanPos)) {
            distance += max(calcRayRangeInSquare(scanPos, direction, OccupancyGrid::BlockSize).y, 0.0f) + SquareTransitionEpsilon;
            continue;
        }
        auto range = calcRayRangeInSquare(scanPos, direction, OccupancyGrid::TileSize);
        auto segmentStart = distance + range.x - SolidHitRadius;

        // A found solid lies at most one scan step before the examined segment
        if (bestMatch != nullptr && isBeyondBestMatch(bestMatch, segmentStart - SolidScanStep)) {
            return -1.0f;
        }
        if (isSolidNearSegment(data, scanPos, direction, range)) {
            auto solidDistance = findSolidInSegment(
                data, origin, direction, max(seedDistance, segmentStart), min(endRadius, distance + range.y + SolidHitRadius), numRefinements);
            if (solidDistance >= 0) {
                // A solid closer than the minimum range blocks the ray
                return solidDistance >= startRadius ? solidDistance : -1.0f;
            }
        }
        distance += max(range.y, 0.0f) + SquareTransitionEpsilon;
    }
    return -1.0f;
}

// The best match is the minimum of all rays, so a ray that is already further away than the best match cannot improve it
__inline__ __device__ bool SensorProcessor::isBeyondBestMatch(uint64_t const* bestMatch, float distance)
{
    auto bestDistanceEncoded = *reinterpret_cast<volatile uint64_t const*>(bestMatch) >> 48;
    return distance * DistanceScale >= toFloat(bestDistanceEncoded + 1);
}

__inline__ __device__ uint64_t SensorProcessor::pack(float distance, float angle, float density, uint16_t misc)
{
    uint32_t angleEncoded = convertAngleToUint16(angle);
    uint32_t densityEncoded = static_cast<uint32_t>(min(65535.0f, density * 100));
    uint64_t distanceEncoded = static_cast<uint64_t>(min(65535.0f, distance * DistanceScale));
    return distanceEncoded << 48 | static_cast<uint64_t>(densityEncoded) << 32 | static_cast<uint64_t>(angleEncoded) << 16 | static_cast<uint64_t>(misc);
}

__inline__ __device__ void SensorProcessor::unpack(float& distance, float& angle, float& density, uint16_t& misc, uint64_t bytes)
{
    distance = toFloat(bytes >> 48) / DistanceScale;
    density = toFloat((bytes >> 32) & 0xFFFF) / 100;
    angle = convertUint16ToAngle(static_cast<int16_t>((bytes >> 16) & 0xFFFF));
    misc = static_cast<int16_t>(bytes & 0xFFFF);
}

__inline__ __device__ void SensorProcessor::writeSignal(NeuralActivity& neuralActivity, float angle, float density, float distance)
{
    neuralActivity.signals[Channels::SensorFoundResult] = 1;                              // Something found
    neuralActivity.signals[Channels::SensorAngle] = angle / 180.0f;                       // Angle: between -1.0 and 1.0
    neuralActivity.signals[Channels::SensorMass] = min(1.0f, density);                    // Normalized density (1.0 = 64 cells in 8x8 region)
    neuralActivity.signals[Channels::SensorDistance] = 1.0f - min(1.0f, distance / 256);  // Distance: 1 = close, 0 = far away
}

__inline__ __device__ uint16_t SensorProcessor::convertAngleToUint16(float angle)
{
    angle = Math::getNormalizedAngle(angle, -180.0f);
    int result = static_cast<int>(angle / 180.0f * 32768.0f);
    return static_cast<uint16_t>(result);
}

__inline__ __device__ float SensorProcessor::convertUint16ToAngle(uint16_t b)
{
    // 0 to 32767 => 0 to 179 degree
    // 32768 to 65535 => -179 to 0 degree
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
