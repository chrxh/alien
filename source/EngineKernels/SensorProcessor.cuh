#pragma once

#include "SimulationData.cuh"

class SensorProcessor
{
public:
    __inline__ __device__ static void process(SimulationData& data, SimulationStatistics& statistics);

private:
    static uint64_t constexpr NoMatch = 0xffffffffffffffff;
    static int constexpr NumScanRays = 64;
    static int constexpr RelocationSearchRadius = 32;  // Search grid is (2*radius) x (2*radius) = 64x64
    static float constexpr ScanStep = 8.0f;
    static int constexpr MaxSameNearCreatureCells = 9 * 9;
    static float constexpr RayBlockingTestLength = 10.0f;

    static float constexpr SolidScanStep = 1.0f;
    static float constexpr SolidHitRadius = 0.75f;
    static int constexpr NumSolidRefinements = 6;
    static int constexpr NumOcclusionRefinements = 2;  // Occlusion needs a lower precision than the distance of a detected solid
    static float constexpr SlotTransitionEpsilon = 0.05f;

    static float constexpr DistanceScale = 64.0f;  // Fixed-point resolution of the packed distance, covers distances up to 1023

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

    __inline__ __device__ static uint64_t matchEnergy(SimulationData& data, float minDensity, float2 const& scanPos, float2 const& delta, float distance);
    __inline__ __device__ static uint64_t
    matchFreeCell(SimulationData& data, float minDensity, uint16_t restrictToColors, float2 const& scanPos, float2 const& delta, float distance);
    __inline__ __device__ static uint64_t matchCreature(SimulationData& data, Object* object, float2 const& scanPos, float2 const& delta, float distance);
    __inline__ __device__ static uint64_t
    matchLastMatchedCreature(SimulationData& data, Object* object, float2 const& scanPos, float2 const& delta, float distance);

    __inline__ __device__ static float calcRayAngle(int rayIdx, float seedAngle);
    __inline__ __device__ static float calcAbsAngle(float2 const& delta);
    __inline__ __device__ static bool isRayBlockedByCreatureConnections(ScanState const& state, float2 const& rayOrigin, float angle);

    __inline__ __device__ static bool isRayBlockedBySolid(SimulationData& data, float2 const& rayOrigin, float angle, float distance);
    __inline__ __device__ static float calcFreeRayLength(SimulationData& data, float2 const& rayOrigin, float angle, float maxLength);
    __inline__ __device__ static bool isSolidNearPosition(SimulationData& data, float2 const& pos, float2 const& direction, float distance);
    __inline__ __device__ static float2 calcRayRangeInSlot(float2 const& scanPos, float2 const& direction);
    __inline__ __device__ static bool mayContainSolid(SimulationData& data, float2 const& scanPos, float2 const& direction, float2 const& range);
    __inline__ __device__ static float
    findSolidInSegment(SimulationData& data, float2 const& origin, float2 const& direction, float startDistance, float endDistance, int numRefinements);
    __inline__ __device__ static float
    findSolidAlongRay(SimulationData& data, float2 const& origin, float angle, float startRadius, float endRadius, float seedDistance, int numRefinements);

    __inline__ __device__ static uint64_t pack(float distance, float angle, float density, uint16_t misc = 0);
    __inline__ __device__ static void unpack(float& distance, float& angle, float& density, uint16_t& misc, uint64_t bytes);

    __inline__ __device__ static void writeSignal(NeuralActivity& neuralActivity, float angle, float density, float distance);

    __inline__ __device__ static uint16_t convertAngleToUint16(float angle);
    __inline__ __device__ static float convertUint16ToAngle(uint16_t b);

    __inline__ __device__ static float calcCreatureDensityFromNumCells(uint32_t numCells);
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
    auto enableRelocationScan = object->typeData.cell.neuralActivity.signals[Channels::SensorWithRelocationScan] < -NEAR_ZERO;
    if (enableRelocationScan && object->typeData.cell.cellTypeData.sensor.lastMatchAvailable) {
        relocateLastMatch(data, statistics, object);
    } else {
        initialScan(data, statistics, object);
    }
}

__inline__ __device__ void SensorProcessor::initialScan(SimulationData& data, SimulationStatistics& statistics, Object* object)
{
    __shared__ ScanState state;

    if (threadIdx.x == 0) {
        state.lookupResult = NoMatch;
        state.seedAngle = data.primaryNumberGen.random(360.0f);

        data.objectMap.getMatchingObjects(
            state.nearSameCreatureCells,
            MaxSameNearCreatureCells,
            state.numNearSameCreatureCells,
            object->pos,
            4.0f,
            object->detached(),
            [&](Object* const& otherObject) {
                return otherObject->type == ObjectType_Cell && object->typeData.cell.isSameCreature(&otherObject->typeData.cell);
            });
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
        auto solidDistance =
            findSolidAlongRay(data, object->pos, angle, startRadius, endRadius, data.primaryNumberGen.random(0, SolidScanStep), NumSolidRefinements);
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
        data.objectMap.correctPosition(scanPos);

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
    auto const& densityMap = data.preprocessedSimulationData.densityMap;

    // Each thread processes multiple rays if blockDim.x < NumScanRays
    for (int rayIdx = threadIdx.x; rayIdx < NumScanRays; rayIdx += blockDim.x) {
        auto angle = calcRayAngle(rayIdx, state.seedAngle);
        if (isRayBlockedByCreatureConnections(state, object->pos, angle)) {
            continue;
        }

        auto freeLength = endRadius;
        auto direction = Math::unitVectorOfAngle(angle);

        for (float distance = data.primaryNumberGen.random(0, ScanStep); distance <= freeLength; distance += ScanStep) {
            auto delta = direction * distance;
            auto scanPos = data.objectMap.getCorrectedPosition(object->pos + delta);

            if (distance > startRadius) {
                auto matchInfo = getMatchInfo(scanPos, delta, distance);
                if (matchInfo != NoMatch) {
                    if (!isRayBlockedBySolid(data, object->pos, angle, distance)) {
                        alienAtomicMin64(&state.lookupResult, matchInfo);
                    }
                    break;
                }
            }

            // A solid in the density map slot does not necessarily lie on the ray
            if (densityMap.getSolidDensity(scanPos) > 0) {
                auto range = calcRayRangeInSlot(scanPos, direction);
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
    auto delta = data.objectMap.getCorrectedDirection(scanPos - object->pos);
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
        data.objectMap.correctPosition(matchPos);

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

__inline__ __device__ uint64_t SensorProcessor::matchEnergy(SimulationData& data, float minDensity, float2 const& scanPos, float2 const& delta, float distance)
{
    auto density = data.preprocessedSimulationData.densityMap.getEnergyParticleDensity(scanPos);
    if (density >= minDensity) {
        return pack(distance, calcAbsAngle(delta), density);
    }
    return NoMatch;
}

__inline__ __device__ uint64_t
SensorProcessor::matchFreeCell(SimulationData& data, float minDensity, uint16_t restrictToColors, float2 const& scanPos, float2 const& delta, float distance)
{
    auto density = data.preprocessedSimulationData.densityMap.getFreeCellDensity(scanPos, restrictToColors);
    if (density >= minDensity) {
        return pack(distance, calcAbsAngle(delta), density);
    }
    return NoMatch;
}

__inline__ __device__ uint64_t SensorProcessor::matchCreature(SimulationData& data, Object* object, float2 const& scanPos, float2 const& delta, float distance)
{
    auto& cell = object->typeData.cell;
    auto const& minNumCells = cell.cellTypeData.sensor.modeData.detectCreature.minNumCells;
    auto const& maxNumCells = cell.cellTypeData.sensor.modeData.detectCreature.maxNumCells;
    auto const& restrictToColors = cell.cellTypeData.sensor.modeData.detectCreature.restrictToColors;
    auto const& restrictToLineage = cell.cellTypeData.sensor.modeData.detectCreature.restrictToLineage;

    auto records = data.objectMap.getRecords();
    int otherIndex = data.objectMap.getFirstIndex(scanPos);
    while (otherIndex >= 0) {
        auto const& otherRecord = records[otherIndex];
        auto otherObject = otherRecord.self;
        // Check if this cell is part of a creature (not solid or free object)
        if (otherObject->type == ObjectType_Cell && !cell.isSameCreature(&otherObject->typeData.cell)) {
            bool matches = true;

            if (!((restrictToColors >> otherObject->color) & 1)) {
                matches = false;
            }
            if (matches && minNumCells > 0 && otherObject->typeData.cell.creature->numCells < minNumCells) {
                matches = false;
            }
            if (matches && maxNumCells > 0 && otherObject->typeData.cell.creature->numCells > maxNumCells) {
                matches = false;
            }
            if (matches && restrictToLineage != LineageRestriction_No) {
                if (restrictToLineage == LineageRestriction_RelatedLineage) {
                    if (!cell.creature->isSameLineage(otherObject->typeData.cell.creature)) {
                        matches = false;
                    }
                } else if (restrictToLineage == LineageRestriction_UnrelatedLineage) {
                    if (cell.creature->isSameLineage(otherObject->typeData.cell.creature)) {
                        matches = false;
                    }
                }
            }

            if (matches) {
                uint16_t creatureIdPart = static_cast<uint16_t>(otherObject->typeData.cell.creature->id & 0xFFFF);
                float density = calcCreatureDensityFromNumCells(otherObject->typeData.cell.creature->numCells);
                return pack(distance, calcAbsAngle(delta), density, creatureIdPart);
            }
        }
        otherIndex = otherRecord.nextObjectIndex;
    }
    return NoMatch;
}

__inline__ __device__ uint64_t
SensorProcessor::matchLastMatchedCreature(SimulationData& data, Object* object, float2 const& scanPos, float2 const& delta, float distance)
{
    auto const& sensor = object->typeData.cell.cellTypeData.sensor;
    auto records = data.objectMap.getRecords();
    int otherIndex = data.objectMap.getFirstIndex(scanPos);
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
    auto rayEnd = rayOrigin + Math::unitVectorOfAngle(angle) * RayBlockingTestLength;
    for (int i = 0; i < state.numNearSameCreatureCells; ++i) {
        auto nearObject = state.nearSameCreatureCells[i];
        for (int j = 0, k = nearObject->numConnections; j < k; ++j) {
            auto& connectedNearObject = nearObject->connections[j].object;
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

// Solids behind the ray origin are ignored, otherwise a solid touching the sensor would block every ray
__inline__ __device__ bool SensorProcessor::isSolidNearPosition(SimulationData& data, float2 const& pos, float2 const& direction, float distance)
{
    auto records = data.objectMap.getRecords();
    int2 const minCell{floorInt(pos.x - SolidHitRadius), floorInt(pos.y - SolidHitRadius)};
    int2 const maxCell{floorInt(pos.x + SolidHitRadius), floorInt(pos.y + SolidHitRadius)};
    for (int cellY = minCell.y; cellY <= maxCell.y; ++cellY) {
        for (int cellX = minCell.x; cellX <= maxCell.x; ++cellX) {
            int2 cell{cellX, cellY};
            data.objectMap.correctPosition(cell);
            auto index = data.objectMap.getFirstIndex(cell);
            for (int level = 0; level < 10 && index >= 0; ++level) {
                auto const& record = records[index];
                if (record.type == ObjectType_Solid) {
                    auto delta = record.self->pos - pos;
                    data.objectMap.correctDirection(delta);
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

// Distances relative to the scan position at which the ray enters and leaves the density map slot of the scan position
__inline__ __device__ float2 SensorProcessor::calcRayRangeInSlot(float2 const& scanPos, float2 const& direction)
{
    auto slotSize = toFloat(DensityMap::SlotSize);

    auto enter = -2.0f * slotSize;
    auto leave = 2.0f * slotSize;
    auto clip = [&](float position, float directionComponent) {
        if (fabsf(directionComponent) > NEAR_ZERO) {
            auto slotStart = floorf(position / slotSize) * slotSize;
            auto invDirectionComponent = 1.0f / directionComponent;
            auto t1 = (slotStart - position) * invDirectionComponent;
            auto t2 = (slotStart + slotSize - position) * invDirectionComponent;
            enter = max(enter, min(t1, t2));
            leave = min(leave, max(t1, t2));
        }
    };
    clip(scanPos.x, direction.x);
    clip(scanPos.y, direction.y);
    return {enter, leave};
}

// The density map slots that are touched by the part of the ray inside a slot, including the margin for the hit radius
__inline__ __device__ bool SensorProcessor::mayContainSolid(SimulationData& data, float2 const& scanPos, float2 const& direction, float2 const& range)
{
    auto const& densityMap = data.preprocessedSimulationData.densityMap;
    auto slotSize = toFloat(DensityMap::SlotSize);

    auto start = scanPos + direction * range.x;
    auto end = scanPos + direction * range.y;
    auto minSlotX = floorInt((min(start.x, end.x) - SolidHitRadius) / slotSize);
    auto maxSlotX = floorInt((max(start.x, end.x) + SolidHitRadius) / slotSize);
    auto minSlotY = floorInt((min(start.y, end.y) - SolidHitRadius) / slotSize);
    auto maxSlotY = floorInt((max(start.y, end.y) + SolidHitRadius) / slotSize);
    for (int slotY = minSlotY; slotY <= maxSlotY; ++slotY) {
        for (int slotX = minSlotX; slotX <= maxSlotX; ++slotX) {
            auto slotCenter = data.objectMap.getCorrectedPosition({(toFloat(slotX) + 0.5f) * slotSize, (toFloat(slotY) + 0.5f) * slotSize});
            if (densityMap.getSolidDensity(slotCenter) > 0) {
                return true;
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
    auto isHit = [&](float distance) {
        return isSolidNearPosition(data, data.objectMap.getCorrectedPosition(origin + direction * distance), direction, distance);
    };

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

// The ray is followed from density map slot to slot, only slots with solids in reach of the ray are examined in detail
__inline__ __device__ float SensorProcessor::findSolidAlongRay(
    SimulationData& data,
    float2 const& origin,
    float angle,
    float startRadius,
    float endRadius,
    float seedDistance,
    int numRefinements)
{
    auto direction = Math::unitVectorOfAngle(angle);

    auto distance = seedDistance;
    while (distance <= endRadius) {
        auto scanPos = data.objectMap.getCorrectedPosition(origin + direction * distance);
        auto range = calcRayRangeInSlot(scanPos, direction);
        if (mayContainSolid(data, scanPos, direction, range)) {
            auto solidDistance = findSolidInSegment(
                data,
                origin,
                direction,
                max(seedDistance, distance + range.x - SolidHitRadius),
                min(endRadius, distance + range.y + SolidHitRadius),
                numRefinements);
            if (solidDistance >= 0) {
                // A solid closer than the minimum range blocks the ray
                return solidDistance >= startRadius ? solidDistance : -1.0f;
            }
        }
        distance += max(range.y, 0.0f) + SlotTransitionEpsilon;
    }
    return -1.0f;
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
