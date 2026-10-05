#pragma once

#include "NeuronProcessor.cuh"
#include "ObjectConnectionProcessor.cuh"
#include "SensorScans.cuh"
#include "SimulationData.cuh"
#include "SimulationStatistics.cuh"

// The domain code is compiled into a separate sensor kernel, so that it does not raise the register usage of a simulation that is not
// split into domains
class SensorProcessor
{
public:
    template <bool Decomposed>
    __inline__ __device__ static void process(SimulationData& data, SimulationStatistics& statistics);

    // Domain decomposition: continuations of the own rays, segments of the rays of other domains and the combination of the results
    __inline__ __device__ static void processContinuations(SimulationData& data);
    __inline__ __device__ static void processScanRequests(SimulationData& data);
    __inline__ __device__ static void addScanResponse(SimulationData& data, SensorScanResponse const& response, int rayIdx);
    __inline__ __device__ static void publishPendingScan(SimulationData& data, Object* object, PendingSensorScan const& scan);

private:
    static int constexpr MaxSameNearCreatureCells = 9 * 9;

    struct ScanState
    {
        uint64_t lookupResult;
        Object* nearSameCreatureCells[MaxSameNearCreatureCells];
        int numNearSameCreatureCells;
        SensorScanParams params;

        // Rays that leave the region of the domain without a result are continued by the domains whose strips they reach
        int numContinuations;
        float continuationStarts[SensorScan::NumRays];  // First sample behind the region, NoContinuation for the other rays
        float continuationEnds[SensorScan::NumRays];
    };

    template <bool Decomposed>
    __inline__ __device__ static void processCell(SimulationData& data, SimulationStatistics& statistics, Object* object);
    template <bool Decomposed>
    __inline__ __device__ static void processDetection(SimulationData& data, SimulationStatistics& statistics, Object* object);

    template <bool Decomposed>
    __inline__ __device__ static void initialScan(SimulationData& data, SimulationStatistics& statistics, Object* object);
    __inline__ __device__ static void initScanParams(SimulationData& data, Object* object, SensorScanType type, SensorScanParams& params);
    template <bool Decomposed>
    __inline__ __device__ static void scanSolid(SimulationData& data, ScanState& state);
    template <bool Decomposed>
    __inline__ __device__ static void scanEnergy(SimulationData& data, ScanState& state);
    template <bool Decomposed>
    __inline__ __device__ static void scanFreeCell(SimulationData& data, ScanState& state);
    template <bool Decomposed>
    __inline__ __device__ static void scanCreature(SimulationData& data, ScanState& state);
    __inline__ __device__ static void scanCreatureNearRange(SimulationData& data, ScanState& state);
    template <bool Decomposed, typename MatchFunc>
    __inline__ __device__ static void scanRays(SimulationData& data, ScanState& state, MatchFunc const& getMatchInfo);

    template <bool Decomposed>
    __inline__ __device__ static void relocateLastMatch(SimulationData& data, SimulationStatistics& statistics, Object* object);
    __inline__ __device__ static uint64_t getRelocationMatchInfo(SimulationData& data, SensorScanParams const& params, float2 const& scanPos);

    __inline__ __device__ static void publishMatch(SimulationData& data, Object* object, uint64_t lookupResult, float2 const& origin, float referenceAngle);
    __inline__ __device__ static void publishNoMatch(Object* object);
    __inline__ __device__ static float calcReferenceAngle(SimulationData& data, Object* object);

    __inline__ __device__ static float calcRegionEnd(SimulationData& data, float2 const& origin, float2 const& direction);
    __inline__ __device__ static void addContinuation(ScanState& state, int rayIdx, float start, float end);
    __inline__ __device__ static bool deferContinuations(SimulationData& data, Object* object, ScanState const& state);
    __inline__ __device__ static void processContinuation(SimulationData& data, SensorContinuation& continuation);
    __inline__ __device__ static bool requestContinuations(SimulationData& data, SensorContinuation& continuation);
    __inline__ __device__ static bool requestRelocation(SimulationData& data, Object* object);
    template <typename Func>
    __inline__ __device__ static void
    forEachContinuationSegment(SimulationData& data, SensorScanParams const& params, int rayIdx, float start, float end, Func const& func);
    __inline__ __device__ static PendingSensorScan*
    createPendingScan(SimulationData& data, Object* object, SensorScanParams const& params, uint64_t bestMatch, uint64_t& index);
    __inline__ __device__ static SensorScanRequest*
    createScanRequest(SimulationData& data, Object* object, SensorScanParams const& params, int targetDomain, uint64_t pendingIndex, uint64_t bestMatch);
    __inline__ __device__ static void processScanRequest(SimulationData& data, SensorScanRequest const& request);

    struct SegmentResult
    {
        uint64_t match = NoMatch;  // First match that no solid on the segment occludes
        float occlusionDistance = SensorScan::NoOcclusion;
    };
    __inline__ __device__ static SegmentResult
    scanSegment(SimulationData& data, SensorScanParams const& params, uint64_t const* bestMatch, int rayIdx, float start, float end);
    template <typename MatchFunc>
    __inline__ __device__ static SegmentResult scanSegment(
        SimulationData& data,
        SensorScanParams const& params,
        uint64_t const* bestMatch,
        int rayIdx,
        float start,
        float end,
        MatchFunc const& getMatchInfo);
    __inline__ __device__ static void relocateForRequest(SimulationData& data, SensorScanRequest const& request, SensorScanResponse& response);

    __inline__ __device__ static uint64_t matchEnergy(SimulationData& data, float minDensity, float2 const& scanPos, float2 const& delta, float distance);
    __inline__ __device__ static uint64_t
    matchFreeCell(SimulationData& data, float minDensity, uint16_t restrictToColors, float2 const& scanPos, float2 const& delta, float distance);
    __inline__ __device__ static uint64_t
    matchCreature(SimulationData& data, SensorScanParams const& params, float2 const& scanPos, float2 const& delta, float distance);
    __inline__ __device__ static uint64_t
    matchLastMatchedCreature(SimulationData& data, SensorScanParams const& params, float2 const& scanPos, float2 const& delta, float distance);

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
        float endRadius,
        float seedDistance,
        int numRefinements,
        uint64_t const* bestMatch = nullptr);
    __inline__ __device__ static bool isBeyondBestMatch(uint64_t const* bestMatch, float distance);

    __inline__ __device__ static uint64_t pack(float distance, float angle, float density, uint16_t misc = 0);
    __inline__ __device__ static void unpack(float& distance, float& angle, float& density, uint16_t& misc, uint64_t bytes);
    __inline__ __device__ static float unpackDistance(uint64_t bytes);

    __inline__ __device__ static void writeSignal(NeuralActivity& neuralActivity, float angle, float density, float distance);

    __inline__ __device__ static uint16_t convertAngleToUint16(float angle);
    __inline__ __device__ static float convertUint16ToAngle(uint16_t b);

    __inline__ __device__ static float calcCreatureDensityFromNumCells(uint32_t numCells);

    static uint64_t constexpr NoMatch = 0xffffffffffffffff;
    static int constexpr NumScanRays = SensorScan::NumRays;
    static int constexpr RelocationSearchRadius = 32;  // Search grid is (2*radius) x (2*radius) = 64x64
    static float constexpr ScanStep = 8.0f;
    static float constexpr RayBlockingTestLength = 10.0f;

    static float constexpr SolidScanStep = 1.0f;
    static float constexpr SolidHitRadius = 0.75f;
    static int constexpr NumSolidRefinements = 6;
    static int constexpr NumOcclusionRefinements = 2;  // Occlusion needs a lower precision than the distance of a detected solid
    static float constexpr SquareTransitionEpsilon = 0.05f;

    static float constexpr DistanceScale = 64.0f;  // Fixed-point resolution of the packed distance, covers distances up to 1023

    static float constexpr StripTransitionEpsilon = 0.05f;
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

template <bool Decomposed>
__inline__ __device__ void SensorProcessor::process(SimulationData& data, SimulationStatistics& statistics)
{
    auto& operations = data.cellTypeOperations[CellType_Sensor];
    auto partition = calcBlockPartition(operations.getNumEntries());
    for (int i = partition.startIndex; i <= partition.endIndex; ++i) {
        processCell<Decomposed>(data, statistics, operations.at(i).object);
    }
}

__inline__ __device__ void SensorProcessor::processContinuations(SimulationData& data)
{
    auto& continuations = data.sensorContinuations;
    auto partition = calcBlockPartition(continuations.getNumEntries());
    for (int i = partition.startIndex; i <= partition.endIndex; ++i) {
        processContinuation(data, continuations.at(i));
    }
}

__inline__ __device__ void SensorProcessor::processScanRequests(SimulationData& data)
{
    auto& requests = data.receivedSensorScanRequests;
    auto partition = calcBlockPartition(requests.getNumEntries());
    for (int i = partition.startIndex; i <= partition.endIndex; ++i) {
        processScanRequest(data, requests.at(i));
    }
}

__inline__ __device__ void SensorProcessor::addScanResponse(SimulationData& data, SensorScanResponse const& response, int rayIdx)
{
    auto& scans = data.pendingSensorScans[response.timestep % 2];
    if (response.pendingIndex >= scans.getNumEntries()) {
        return;
    }
    auto& scan = scans.at(response.pendingIndex);
    if (!scan.active || scan.sensorId != response.sensorId || scan.timestep != response.timestep) {
        return;
    }
    alienAtomicMin64(&scan.matches[rayIdx], response.matches[rayIdx]);
    atomicMin(&scan.occlusionDistances[rayIdx], __float_as_int(response.occlusionDistances[rayIdx]));
}

// A match of a ray counts if no solid on any segment of the ray occludes it
__inline__ __device__ void SensorProcessor::publishPendingScan(SimulationData& data, Object* object, PendingSensorScan const& scan)
{
    auto const& params = scan.params;
    auto result = scan.bestMatch;
    for (int rayIdx = 0; rayIdx < NumScanRays; ++rayIdx) {
        auto occlusionDistance = __int_as_float(scan.occlusionDistances[rayIdx]);
        auto candidate = NoMatch;
        if (params.mode == SensorMode_DetectSolid) {
            if (occlusionDistance >= params.startRadius && occlusionDistance <= params.endRadius) {
                candidate = pack(occlusionDistance, Math::getNormalizedAngle(calcRayAngle(rayIdx, params.seedAngle), -180.0f), 1.0f);
            }
        } else {
            auto match = scan.matches[rayIdx];
            if (match != NoMatch && unpackDistance(match) < occlusionDistance) {
                candidate = match;
            }
        }
        if (candidate < result) {
            result = candidate;
        }
    }
    if (result != NoMatch) {
        publishMatch(data, object, result, params.origin, scan.referenceAngle);
    } else {
        publishNoMatch(object);
    }
}

template <bool Decomposed>
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
        processDetection<Decomposed>(data, statistics, object);
    }
}

template <bool Decomposed>
__inline__ __device__ void SensorProcessor::processDetection(SimulationData& data, SimulationStatistics& statistics, Object* object)
{
    auto enableRelocationScan = object->typeData.cell.neuralActivity.signals[Channels::SensorWithRelocationScan] < -NEAR_ZERO;
    if (enableRelocationScan && object->typeData.cell.cellTypeData.sensor.lastMatchAvailable) {
        relocateLastMatch<Decomposed>(data, statistics, object);
    } else {
        initialScan<Decomposed>(data, statistics, object);
    }
}

template <bool Decomposed>
__inline__ __device__ void SensorProcessor::initialScan(SimulationData& data, SimulationStatistics& statistics, Object* object)
{
    __shared__ ScanState state;

    if (threadIdx.x == 0) {
        state.lookupResult = NoMatch;
        initScanParams(data, object, SensorScanType::Rays, state.params);
        state.params.seedAngle = data.primaryNumberGen.random(360.0f);
        state.numNearSameCreatureCells = 0;
        state.numContinuations = 0;
    }
    if constexpr (Decomposed) {
        for (int rayIdx = threadIdx.x; rayIdx < NumScanRays; rayIdx += blockDim.x) {
            state.continuationStarts[rayIdx] = SensorScan::NoContinuation;
        }
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

    switch (state.params.mode) {
    case SensorMode_DetectSolid:
        scanSolid<Decomposed>(data, state);
        break;
    case SensorMode_DetectEnergy:
        scanEnergy<Decomposed>(data, state);
        break;
    case SensorMode_DetectFreeCell:
        scanFreeCell<Decomposed>(data, state);
        break;
    case SensorMode_DetectCreature:
        scanCreature<Decomposed>(data, state);
        break;
    }
    __syncthreads();

    if constexpr (Decomposed) {
        if (state.numContinuations > 0 && deferContinuations(data, object, state)) {
            return;
        }
    }

    if (threadIdx.x == 0) {
        if (state.lookupResult != NoMatch) {
            publishMatch(data, object, state.lookupResult, object->pos, calcReferenceAngle(data, object));
        } else {
            publishNoMatch(object);
        }
    }
}

__inline__ __device__ void SensorProcessor::initScanParams(SimulationData& data, Object* object, SensorScanType type, SensorScanParams& params)
{
    auto const& cell = object->typeData.cell;
    auto const& sensor = cell.cellTypeData.sensor;
    params.type = type;
    params.mode = sensor.mode;
    params.origin = object->pos;
    params.startRadius = toFloat(sensor.minRange);
    params.endRadius = min(cudaSimulationParameters.sensorRadius.value[object->color], toFloat(sensor.maxRange));
    params.seedAngle = 0;
    params.minDensity = 0;
    params.restrictToColors = 0;
    params.lastMatchCreatureIdPart = sensor.lastMatch.creatureIdPart;
    params.minNumCells = 0;
    params.maxNumCells = 0;
    params.restrictToLineage = LineageRestriction_No;
    params.lineageId = cell.creature->lineageId;
    params.creatureId = cell.creature->id;
    params.relocationCenter = sensor.lastMatch.pos;
    switch (sensor.mode) {
    case SensorMode_DetectEnergy:
        params.minDensity = sensor.modeData.detectEnergy.minDensity;
        break;
    case SensorMode_DetectFreeCell:
        params.minDensity = sensor.modeData.detectFreeCell.minDensity;
        params.restrictToColors = sensor.modeData.detectFreeCell.restrictToColors;
        break;
    case SensorMode_DetectCreature:
        params.restrictToColors = sensor.modeData.detectCreature.restrictToColors;
        params.minNumCells = sensor.modeData.detectCreature.minNumCells;
        params.maxNumCells = sensor.modeData.detectCreature.maxNumCells;
        params.restrictToLineage = sensor.modeData.detectCreature.restrictToLineage;
        break;
    }
}

template <bool Decomposed>
__inline__ __device__ void SensorProcessor::scanSolid(SimulationData& data, ScanState& state)
{
    auto const& params = state.params;
    for (int rayIdx = threadIdx.x; rayIdx < NumScanRays; rayIdx += blockDim.x) {
        auto angle = calcRayAngle(rayIdx, params.seedAngle);
        if (isRayBlockedByCreatureConnections(state, params.origin, angle)) {
            continue;
        }
        auto regionEnd = Decomposed ? calcRegionEnd(data, params.origin, Math::unitVectorOfAngle(angle)) : FLT_MAX;
        auto scanEnd = min(params.endRadius, regionEnd);
        auto solidDistance =
            findSolidAlongRay(data, params.origin, angle, scanEnd, data.primaryNumberGen.random(0, SolidScanStep), NumSolidRefinements, &state.lookupResult);

        // A solid closer than the minimum range blocks the ray
        if (solidDistance >= params.startRadius) {
            alienAtomicMin64(&state.lookupResult, pack(solidDistance, Math::getNormalizedAngle(angle, -180.0f), 1.0f));
        } else if constexpr (Decomposed) {
            if (solidDistance < 0 && scanEnd < params.endRadius) {
                addContinuation(state, rayIdx, regionEnd, params.endRadius);
            }
        }
    }
}

template <bool Decomposed>
__inline__ __device__ void SensorProcessor::scanEnergy(SimulationData& data, ScanState& state)
{
    auto const& params = state.params;
    scanRays<Decomposed>(data, state, [&](float2 const& scanPos, float2 const& delta, float distance) {
        return matchEnergy(data, params.minDensity, scanPos, delta, distance);
    });
}

template <bool Decomposed>
__inline__ __device__ void SensorProcessor::scanFreeCell(SimulationData& data, ScanState& state)
{
    auto const& params = state.params;
    scanRays<Decomposed>(data, state, [&](float2 const& scanPos, float2 const& delta, float distance) {
        return matchFreeCell(data, params.minDensity, params.restrictToColors, scanPos, delta, distance);
    });
}

template <bool Decomposed>
__inline__ __device__ void SensorProcessor::scanCreature(SimulationData& data, ScanState& state)
{
    scanCreatureNearRange(data, state);
    __syncthreads();

    if (state.lookupResult == NoMatch) {
        auto const& params = state.params;
        scanRays<Decomposed>(
            data, state, [&](float2 const& scanPos, float2 const& delta, float distance) { return matchCreature(data, params, scanPos, delta, distance); });
    }
}

__inline__ __device__ void SensorProcessor::scanCreatureNearRange(SimulationData& data, ScanState& state)
{
    auto const& params = state.params;
    auto nearDistance = toInt(ScanStep);
    int diameter = 2 * nearDistance + 1;
    int totalPositions = diameter * diameter;

    // Each thread scans different positions in parallel
    for (int idx = threadIdx.x; idx < totalPositions; idx += blockDim.x) {
        auto dx = toFloat((idx % diameter) - nearDistance);
        auto dy = toFloat((idx / diameter) - nearDistance);

        auto delta = float2{dx, dy};
        auto distance = Math::length(delta);
        if (distance <= params.startRadius || distance > params.endRadius) {
            continue;
        }
        float2 scanPos = params.origin + delta;
        data.world.correctPosition(scanPos);

        // Check all cells at this position (including overlapping cells)
        auto matchInfo = matchCreature(data, params, scanPos, delta, distance);
        if (matchInfo != NoMatch) {
            auto angle = Math::angleOfVector(delta);
            if (!isRayBlockedByCreatureConnections(state, params.origin, angle) && !isRayBlockedBySolid(data, params.origin, angle, distance)) {
                alienAtomicMin64(&state.lookupResult, matchInfo);
            }
        }
    }
}

template <bool Decomposed, typename MatchFunc>
__inline__ __device__ void SensorProcessor::scanRays(SimulationData& data, ScanState& state, MatchFunc const& getMatchInfo)
{
    auto const& params = state.params;

    // Each thread processes multiple rays if blockDim.x < NumScanRays
    for (int rayIdx = threadIdx.x; rayIdx < NumScanRays; rayIdx += blockDim.x) {
        auto angle = calcRayAngle(rayIdx, params.seedAngle);
        if (isRayBlockedByCreatureConnections(state, params.origin, angle)) {
            continue;
        }

        auto freeLength = params.endRadius;
        auto direction = Math::unitVectorOfAngle(angle);
        [[maybe_unused]] auto regionEnd = Decomposed ? calcRegionEnd(data, params.origin, direction) : FLT_MAX;

        for (float distance = data.primaryNumberGen.random(0, ScanStep); distance <= freeLength && !isBeyondBestMatch(&state.lookupResult, distance);
             distance += ScanStep) {
            if constexpr (Decomposed) {
                if (distance >= regionEnd) {
                    addContinuation(state, rayIdx, distance, freeLength);
                    break;
                }
            }
            auto delta = direction * distance;
            auto scanPos = data.world.getCorrectedPosition(params.origin + delta);

            if (distance > params.startRadius) {
                auto matchInfo = getMatchInfo(scanPos, delta, distance);
                if (matchInfo != NoMatch) {
                    if (!isRayBlockedBySolid(data, params.origin, angle, distance)) {
                        alienAtomicMin64(&state.lookupResult, matchInfo);
                    }
                    break;
                }
            }

            auto range = calcRayRangeInSquare(scanPos, direction, OccupancyGrid::TileSize);
            if (isSolidNearSegment(data, scanPos, direction, range)) {
                auto solidDistance = findSolidInSegment(
                    data,
                    params.origin,
                    direction,
                    max(distance + range.x - SolidHitRadius, 0.0f),
                    min(distance + range.y + SolidHitRadius, params.endRadius),
                    NumOcclusionRefinements);
                if (solidDistance >= 0) {
                    freeLength = min(freeLength, solidDistance);
                }
            }
        }
    }
}

template <bool Decomposed>
__inline__ __device__ void SensorProcessor::relocateLastMatch(SimulationData& data, SimulationStatistics& statistics, Object* object)
{
    if constexpr (Decomposed) {
        if (requestRelocation(data, object)) {
            return;
        }
    }

    // Search grid is (2*RelocationSearchRadius) x (2*RelocationSearchRadius)
    // Each thread handles multiple columns if blockDim.x < (2*RelocationSearchRadius)
    int searchDiameter = 2 * RelocationSearchRadius;

    __shared__ uint64_t lookupResult;
    __shared__ SensorScanParams params;
    if (threadIdx.x == 0) {
        lookupResult = NoMatch;
        initScanParams(data, object, SensorScanType::Relocation, params);
    }
    __syncthreads();

    // Each thread handles multiple columns (deltaX values)
    for (int colIdx = threadIdx.x; colIdx < searchDiameter; colIdx += blockDim.x) {
        int deltaX = colIdx - RelocationSearchRadius;

        for (int deltaY = -RelocationSearchRadius; deltaY < RelocationSearchRadius; ++deltaY) {
            auto scanPos = params.relocationCenter + float2{toFloat(deltaX), toFloat(deltaY)};
            auto matchInfo = getRelocationMatchInfo(data, params, scanPos);
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
            isValid = distance >= params.startRadius && distance <= params.endRadius && !isRayBlockedBySolid(data, object->pos, absAngle, distance);
        }
        if (isValid) {
            publishMatch(data, object, lookupResult, object->pos, calcReferenceAngle(data, object));
        } else {
            publishNoMatch(object);
        }
    }
}

__inline__ __device__ uint64_t SensorProcessor::getRelocationMatchInfo(SimulationData& data, SensorScanParams const& params, float2 const& scanPos)
{
    auto delta = data.world.getCorrectedDirection(scanPos - params.origin);
    auto distance = Math::length(delta);

    switch (params.mode) {
    case SensorMode_DetectEnergy:
        return matchEnergy(data, params.minDensity, scanPos, delta, distance);
    case SensorMode_DetectFreeCell:
        return matchFreeCell(data, params.minDensity, params.restrictToColors, scanPos, delta, distance);
    case SensorMode_DetectCreature:
        return matchLastMatchedCreature(data, params, scanPos, delta, distance);
    }
    return NoMatch;
}

__inline__ __device__ void
SensorProcessor::publishMatch(SimulationData& data, Object* object, uint64_t lookupResult, float2 const& origin, float referenceAngle)
{
    float distance, absAngle, density;
    uint16_t creatureIdPart;
    unpack(distance, absAngle, density, creatureIdPart, lookupResult);

    auto& cell = object->typeData.cell;
    auto relAngle = Math::getNormalizedAngle(absAngle - referenceAngle - cell.frontAngle, -180.0f);
    writeSignal(cell.neuralActivity, relAngle, density, distance);

    // No relocation for solids
    if (cell.cellTypeData.sensor.mode != SensorMode_DetectSolid) {
        auto matchPos = origin + Math::unitVectorOfAngle(absAngle) * distance;
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

__inline__ __device__ float SensorProcessor::calcReferenceAngle(SimulationData& data, Object* object)
{
    return Math::angleOfVector(ObjectConnectionProcessor::calcReferenceDirection(data, object));
}

// The domain scans a ray as long as it has the complete data around the samples, which holds within the halo minus a density slot
__inline__ __device__ float SensorProcessor::calcRegionEnd(SimulationData& data, float2 const& origin, float2 const& direction)
{
    auto const& domain = data.domain;
    return domain.getStripExitDistance(domain.index, origin, direction, domain.layout->haloWidth - toFloat(DensityGrid::SlotSize));
}

__inline__ __device__ void SensorProcessor::addContinuation(ScanState& state, int rayIdx, float start, float end)
{
    state.continuationStarts[rayIdx] = start;
    state.continuationEnds[rayIdx] = end;
    atomicAdd(&state.numContinuations, 1);
}

// Hands the scan over to the kernel that requests the continuations, returns false if the scan has to be published right away
__inline__ __device__ bool SensorProcessor::deferContinuations(SimulationData& data, Object* object, ScanState const& state)
{
    __shared__ SensorContinuation* continuation;
    if (threadIdx.x == 0) {
        continuation = data.sensorContinuations.tryGetNewElement();
        if (continuation) {
            continuation->sensor = object;
            continuation->bestMatch = state.lookupResult;
            continuation->params = state.params;
        }
    }
    __syncthreads();

    if (!continuation) {
        return false;
    }
    for (int rayIdx = threadIdx.x; rayIdx < NumScanRays; rayIdx += blockDim.x) {
        continuation->starts[rayIdx] = state.continuationStarts[rayIdx];
        continuation->ends[rayIdx] = state.continuationEnds[rayIdx];
    }
    return true;
}

__inline__ __device__ void SensorProcessor::processContinuation(SimulationData& data, SensorContinuation& continuation)
{
    if (!requestContinuations(data, continuation) && threadIdx.x == 0) {
        auto object = continuation.sensor;
        if (continuation.bestMatch != NoMatch) {
            publishMatch(data, object, continuation.bestMatch, continuation.params.origin, calcReferenceAngle(data, object));
        } else {
            publishNoMatch(object);
        }
    }
    __syncthreads();
}

// Returns true if the continued rays are scanned by other domains, whose results are combined two sync rounds later. Segments in
// the own strip, which rays reach after leaving the region, are scanned right away.
__inline__ __device__ bool SensorProcessor::requestContinuations(SimulationData& data, SensorContinuation& continuation)
{
    __shared__ uint32_t targetDomains;
    __shared__ PendingSensorScan* pendingScan;
    __shared__ SensorScanRequest* requests[DomainLayout::MaxDomains];

    auto object = continuation.sensor;
    auto const& params = continuation.params;
    auto const& domain = data.domain;
    if (threadIdx.x == 0) {
        targetDomains = 0;
    }
    __syncthreads();

    // Rays that cannot improve the result or are occluded before their continuation are dropped
    for (int rayIdx = threadIdx.x; rayIdx < NumScanRays; rayIdx += blockDim.x) {
        auto start = continuation.starts[rayIdx];
        if (start == SensorScan::NoContinuation) {
            continue;
        }
        auto angle = calcRayAngle(rayIdx, params.seedAngle);
        if (isBeyondBestMatch(&continuation.bestMatch, start)
            || (params.mode != SensorMode_DetectSolid && isRayBlockedBySolid(data, params.origin, angle, start))) {
            continuation.starts[rayIdx] = SensorScan::NoContinuation;
            continue;
        }
        forEachContinuationSegment(
            data, params, rayIdx, start, continuation.ends[rayIdx], [&](int targetDomain, float, float) { atomicOr(&targetDomains, 1u << targetDomain); });
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        pendingScan = nullptr;
        if (targetDomains != 0) {
            uint64_t pendingIndex;
            pendingScan = createPendingScan(data, object, params, continuation.bestMatch, pendingIndex);
            if (pendingScan) {
                pendingScan->active = (targetDomains >> domain.index) & 1;
                for (int targetDomain = 0; targetDomain < domain.numDomains; ++targetDomain) {
                    requests[targetDomain] = nullptr;
                    if (targetDomain != domain.index && ((targetDomains >> targetDomain) & 1)) {
                        requests[targetDomain] = createScanRequest(data, object, params, targetDomain, pendingIndex, continuation.bestMatch);
                        if (requests[targetDomain]) {
                            pendingScan->active = true;
                        }
                    }
                }
                if (!pendingScan->active) {
                    pendingScan = nullptr;
                }
            }
        }
    }
    __syncthreads();

    if (!pendingScan) {
        return false;
    }
    for (int rayIdx = threadIdx.x; rayIdx < NumScanRays; rayIdx += blockDim.x) {
        for (int targetDomain = 0; targetDomain < domain.numDomains; ++targetDomain) {
            if (auto request = requests[targetDomain]) {
                request->segmentStarts[rayIdx] = SensorScan::NoSegment;
            }
        }
        auto start = continuation.starts[rayIdx];
        if (start == SensorScan::NoContinuation) {
            continue;
        }
        forEachContinuationSegment(data, params, rayIdx, start, continuation.ends[rayIdx], [&](int targetDomain, float segmentStart, float segmentEnd) {
            if (targetDomain == domain.index) {
                auto result = scanSegment(data, params, &continuation.bestMatch, rayIdx, segmentStart, segmentEnd);
                pendingScan->matches[rayIdx] = result.match;
                pendingScan->occlusionDistances[rayIdx] = __float_as_int(result.occlusionDistance);
            } else if (auto request = requests[targetDomain]) {
                request->segmentStarts[rayIdx] = segmentStart;
                request->segmentEnds[rayIdx] = segmentEnd;
            }
        });
    }
    return true;
}

// Returns true if the search area lies in the strip of another domain, which then relocates the match
__inline__ __device__ bool SensorProcessor::requestRelocation(SimulationData& data, Object* object)
{
    __shared__ bool isRemote;
    if (threadIdx.x == 0) {
        auto const& domain = data.domain;
        auto center = object->typeData.cell.cellTypeData.sensor.lastMatch.pos;
        isRemote = domain.getDistanceToStrip(domain.index, center.x) > 0;
        if (isRemote) {
            SensorScanParams params;
            initScanParams(data, object, SensorScanType::Relocation, params);

            // The own part of the ray towards the search area
            auto delta = data.world.getCorrectedDirection(center - object->pos);
            auto distance = Math::length(delta);
            auto angle = Math::angleOfVector(delta);
            auto ownPart = min(distance, calcRegionEnd(data, object->pos, Math::unitVectorOfAngle(angle)));
            auto isBlocked = isRayBlockedBySolid(data, object->pos, angle, ownPart);

            uint64_t pendingIndex;
            auto pendingScan = isBlocked ? nullptr : createPendingScan(data, object, params, NoMatch, pendingIndex);
            if (pendingScan) {
                auto request = createScanRequest(data, object, params, domain.getStripOwner(center.x), pendingIndex, NoMatch);
                pendingScan->active = request != nullptr;
            }
            if (!pendingScan || !pendingScan->active) {
                publishNoMatch(object);
            }
        }
    }
    __syncthreads();
    return isRemote;
}

// Calls func(targetDomain, start, end) for the first segment of the continued ray in each strip it crosses. The segments end on
// the sample grid of the ray behind the strips, so that every sample is scanned by the domain of its strip.
template <typename Func>
__inline__ __device__ void
SensorProcessor::forEachContinuationSegment(SimulationData& data, SensorScanParams const& params, int rayIdx, float start, float end, Func const& func)
{
    auto const& domain = data.domain;
    auto direction = Math::unitVectorOfAngle(calcRayAngle(rayIdx, params.seedAngle));
    auto sampleStep = params.mode == SensorMode_DetectSolid ? 0.0f : ScanStep;
    auto sampleOrigin = start;
    uint32_t visitedDomains = 0;
    while (start < end) {
        auto pos = data.world.getCorrectedPosition(params.origin + direction * start);
        auto owner = domain.getStripOwner(pos.x);
        auto stripExit = start + domain.getStripExitDistance(owner, pos, direction, 0);
        auto next = sampleStep > 0 ? sampleOrigin + ceilf((stripExit - sampleOrigin) / sampleStep) * sampleStep : stripExit;
        next = min(max(next, start + max(sampleStep, StripTransitionEpsilon)), end);
        if (!((visitedDomains >> owner) & 1)) {
            visitedDomains |= 1u << owner;
            func(owner, start, next);
        }
        start = next;
    }
}

__inline__ __device__ PendingSensorScan*
SensorProcessor::createPendingScan(SimulationData& data, Object* object, SensorScanParams const& params, uint64_t bestMatch, uint64_t& index)
{
    auto timestep = *data.timestep;
    auto scan = data.pendingSensorScans[timestep % 2].tryGetNewElement(&index);
    if (scan) {
        scan->active = false;
        scan->sensorId = object->id;
        scan->timestep = timestep;
        scan->referenceAngle = calcReferenceAngle(data, object);
        scan->params = params;
        scan->bestMatch = bestMatch;
        for (int rayIdx = 0; rayIdx < NumScanRays; ++rayIdx) {
            scan->matches[rayIdx] = NoMatch;
            scan->occlusionDistances[rayIdx] = __float_as_int(SensorScan::NoOcclusion);
        }
    }
    return scan;
}

__inline__ __device__ SensorScanRequest* SensorProcessor::createScanRequest(
    SimulationData& data,
    Object* object,
    SensorScanParams const& params,
    int targetDomain,
    uint64_t pendingIndex,
    uint64_t bestMatch)
{
    auto request = data.sensorScanRequests.tryGetNewElement();
    if (request) {
        request->requesterDomain = static_cast<uint8_t>(data.domain.index);
        request->targetDomain = static_cast<uint8_t>(targetDomain);
        request->pendingIndex = static_cast<uint32_t>(pendingIndex);
        request->sensorId = object->id;
        request->timestep = *data.timestep;
        request->bestMatch = bestMatch;
        request->params = params;
    }
    return request;
}

__inline__ __device__ void SensorProcessor::processScanRequest(SimulationData& data, SensorScanRequest const& request)
{
    __shared__ SensorScanResponse* response;
    if (threadIdx.x == 0) {
        response = data.sensorScanResponses.tryGetNewElement();
        if (response) {
            response->targetDomain = request.requesterDomain;
            response->pendingIndex = request.pendingIndex;
            response->sensorId = request.sensorId;
            response->timestep = request.timestep;
        }
    }
    __syncthreads();

    if (response) {
        for (int rayIdx = threadIdx.x; rayIdx < NumScanRays; rayIdx += blockDim.x) {
            response->matches[rayIdx] = NoMatch;
            response->occlusionDistances[rayIdx] = SensorScan::NoOcclusion;
        }
        __syncthreads();

        if (request.params.type == SensorScanType::Relocation) {
            relocateForRequest(data, request, *response);
        } else {
            for (int rayIdx = threadIdx.x; rayIdx < NumScanRays; rayIdx += blockDim.x) {
                if (request.segmentStarts[rayIdx] != SensorScan::NoSegment) {
                    auto result = scanSegment(data, request.params, &request.bestMatch, rayIdx, request.segmentStarts[rayIdx], request.segmentEnds[rayIdx]);
                    response->matches[rayIdx] = result.match;
                    response->occlusionDistances[rayIdx] = result.occlusionDistance;
                }
            }
        }
    }
    __syncthreads();
}

__inline__ __device__ SensorProcessor::SegmentResult
SensorProcessor::scanSegment(SimulationData& data, SensorScanParams const& params, uint64_t const* bestMatch, int rayIdx, float start, float end)
{
    switch (params.mode) {
    case SensorMode_DetectSolid: {
        SegmentResult result;
        auto solidDistance = findSolidAlongRay(data, params.origin, calcRayAngle(rayIdx, params.seedAngle), end, start, NumSolidRefinements, bestMatch);
        if (solidDistance >= 0) {
            result.occlusionDistance = solidDistance;
        }
        return result;
    }
    case SensorMode_DetectEnergy:
        return scanSegment(data, params, bestMatch, rayIdx, start, end, [&](float2 const& scanPos, float2 const& delta, float distance) {
            return matchEnergy(data, params.minDensity, scanPos, delta, distance);
        });
    case SensorMode_DetectFreeCell:
        return scanSegment(data, params, bestMatch, rayIdx, start, end, [&](float2 const& scanPos, float2 const& delta, float distance) {
            return matchFreeCell(data, params.minDensity, params.restrictToColors, scanPos, delta, distance);
        });
    case SensorMode_DetectCreature:
        return scanSegment(data, params, bestMatch, rayIdx, start, end, [&](float2 const& scanPos, float2 const& delta, float distance) {
            return matchCreature(data, params, scanPos, delta, distance);
        });
    }
    return SegmentResult();
}

// Reports the first match that no solid on the segment occludes, otherwise the first solid, which occludes the following segments
template <typename MatchFunc>
__inline__ __device__ SensorProcessor::SegmentResult SensorProcessor::scanSegment(
    SimulationData& data,
    SensorScanParams const& params,
    uint64_t const* bestMatch,
    int rayIdx,
    float start,
    float end,
    MatchFunc const& getMatchInfo)
{
    SegmentResult result;
    auto angle = calcRayAngle(rayIdx, params.seedAngle);
    auto direction = Math::unitVectorOfAngle(angle);

    auto freeLength = end;
    for (float distance = start; distance < end && distance <= freeLength; distance += ScanStep) {
        if (isBeyondBestMatch(bestMatch, distance)) {
            return result;
        }
        auto delta = direction * distance;
        auto scanPos = data.world.getCorrectedPosition(params.origin + delta);

        if (distance > params.startRadius) {
            auto matchInfo = getMatchInfo(scanPos, delta, distance);
            if (matchInfo != NoMatch) {
                auto solidDistance = findSolidAlongRay(data, params.origin, angle, distance, start, NumOcclusionRefinements);
                if (solidDistance >= 0 && solidDistance < distance) {
                    result.occlusionDistance = solidDistance;
                } else {
                    result.match = matchInfo;
                }
                return result;
            }
        }

        auto range = calcRayRangeInSquare(scanPos, direction, OccupancyGrid::TileSize);
        if (isSolidNearSegment(data, scanPos, direction, range)) {
            auto solidDistance = findSolidInSegment(
                data,
                params.origin,
                direction,
                max(distance + range.x - SolidHitRadius, start),
                min(distance + range.y + SolidHitRadius, end),
                NumOcclusionRefinements);
            if (solidDistance >= 0) {
                freeLength = min(freeLength, solidDistance);
            }
        }
    }
    auto solidDistance = freeLength < end ? freeLength : findSolidAlongRay(data, params.origin, angle, end, start, NumOcclusionRefinements);
    if (solidDistance >= 0) {
        result.occlusionDistance = solidDistance;
    }
    return result;
}

// The requester checked the part of the ray in its region, the rest is checked for solids in the own region
__inline__ __device__ void SensorProcessor::relocateForRequest(SimulationData& data, SensorScanRequest const& request, SensorScanResponse& response)
{
    auto const& params = request.params;
    int searchDiameter = 2 * RelocationSearchRadius;

    __shared__ uint64_t lookupResult;
    if (threadIdx.x == 0) {
        lookupResult = NoMatch;
    }
    __syncthreads();

    for (int colIdx = threadIdx.x; colIdx < searchDiameter; colIdx += blockDim.x) {
        int deltaX = colIdx - RelocationSearchRadius;
        for (int deltaY = -RelocationSearchRadius; deltaY < RelocationSearchRadius; ++deltaY) {
            auto scanPos = params.relocationCenter + float2{toFloat(deltaX), toFloat(deltaY)};
            auto matchInfo = getRelocationMatchInfo(data, params, scanPos);
            if (matchInfo != NoMatch) {
                alienAtomicMin64(&lookupResult, matchInfo);
                break;
            }
        }
    }
    __syncthreads();

    if (threadIdx.x == 0 && lookupResult != NoMatch) {
        float distance, absAngle, density;
        uint16_t creatureIdPart;
        unpack(distance, absAngle, density, creatureIdPart, lookupResult);
        if (distance >= params.startRadius && distance <= params.endRadius) {
            auto direction = Math::unitVectorOfAngle(absAngle);
            auto matchPos = data.world.getCorrectedPosition(params.origin + direction * distance);
            auto regionLength = data.domain.getStripExitDistance(
                data.domain.index, matchPos, direction * -1.0f, data.domain.layout->haloWidth - toFloat(DensityGrid::SlotSize));
            auto solidDistance = findSolidAlongRay(data, params.origin, absAngle, distance, max(0.0f, distance - regionLength), NumOcclusionRefinements);
            if (solidDistance < 0 || solidDistance >= distance) {
                response.matches[0] = lookupResult;
            }
        }
    }
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

__inline__ __device__ uint64_t
SensorProcessor::matchCreature(SimulationData& data, SensorScanParams const& params, float2 const& scanPos, float2 const& delta, float distance)
{
    auto records = data.objectGrid.getRecords();
    int otherIndex = data.objectGrid.getFirstIndex(scanPos);
    while (otherIndex >= 0) {
        auto const& otherRecord = records[otherIndex];
        auto otherObject = otherRecord.self;
        // Check if this cell is part of a creature (not solid or free object)
        if (otherObject->type == ObjectType_Cell && otherObject->typeData.cell.creature->id != params.creatureId) {
            auto otherCreature = otherObject->typeData.cell.creature;
            bool matches = true;

            if (!((params.restrictToColors >> otherObject->color) & 1)) {
                matches = false;
            }
            if (matches && params.minNumCells > 0 && otherCreature->numCells < params.minNumCells) {
                matches = false;
            }
            if (matches && params.maxNumCells > 0 && otherCreature->numCells > params.maxNumCells) {
                matches = false;
            }
            if (matches && params.restrictToLineage != LineageRestriction_No) {
                if (params.restrictToLineage == LineageRestriction_RelatedLineage) {
                    if (otherCreature->lineageId != params.lineageId) {
                        matches = false;
                    }
                } else if (params.restrictToLineage == LineageRestriction_UnrelatedLineage) {
                    if (otherCreature->lineageId == params.lineageId) {
                        matches = false;
                    }
                }
            }

            if (matches) {
                uint16_t creatureIdPart = static_cast<uint16_t>(otherCreature->id & 0xFFFF);
                float density = calcCreatureDensityFromNumCells(otherCreature->numCells);
                return pack(distance, calcAbsAngle(delta), density, creatureIdPart);
            }
        }
        otherIndex = otherRecord.nextObjectIndex;
    }
    return NoMatch;
}

__inline__ __device__ uint64_t
SensorProcessor::matchLastMatchedCreature(SimulationData& data, SensorScanParams const& params, float2 const& scanPos, float2 const& delta, float distance)
{
    auto records = data.objectGrid.getRecords();
    int otherIndex = data.objectGrid.getFirstIndex(scanPos);
    while (otherIndex >= 0) {
        auto const& otherRecord = records[otherIndex];
        auto otherObject = otherRecord.self;
        if (otherObject->type == ObjectType_Cell && (otherObject->typeData.cell.creature->id & 0xffff) == params.lastMatchCreatureIdPart) {
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
    auto solidDistance = findSolidAlongRay(data, rayOrigin, angle, maxLength, 0.0f, NumOcclusionRefinements);
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
    if (!data.barrierGrid.hasSolidNearBlockOf(pos)) {
        return false;
    }
    auto start = pos + direction * range.x;
    auto end = pos + direction * range.y;
    int2 minPos{floorInt(min(start.x, end.x) - SolidHitRadius), floorInt(min(start.y, end.y) - SolidHitRadius)};
    int2 maxPos{floorInt(max(start.x, end.x) + SolidHitRadius), floorInt(max(start.y, end.y) + SolidHitRadius)};
    return data.barrierGrid.hasSolid(minPos, maxPos);
}

// Solids behind the ray origin are ignored, otherwise a solid touching the sensor would block every ray
__inline__ __device__ bool SensorProcessor::isSolidNearPosition(SimulationData& data, float2 const& pos, float2 const& direction, float distance)
{
    auto records = data.objectGrid.getRecords();
    int2 const minCell{floorInt(pos.x - SolidHitRadius), floorInt(pos.y - SolidHitRadius)};
    int2 const maxCell{floorInt(pos.x + SolidHitRadius), floorInt(pos.y + SolidHitRadius)};
    if (!data.barrierGrid.hasSolid(minCell, maxCell)) {
        return false;
    }
    for (int cellY = minCell.y; cellY <= maxCell.y; ++cellY) {
        for (int cellX = minCell.x; cellX <= maxCell.x; ++cellX) {
            int2 cell{cellX, cellY};
            data.world.correctPosition(cell);
            if (!data.barrierGrid.hasSolid(cell)) {
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
// Only tiles with solids in reach of the ray are examined in detail. Returns the distance of the first solid or a negative value.
__inline__ __device__ float SensorProcessor::findSolidAlongRay(
    SimulationData& data,
    float2 const& origin,
    float angle,
    float endRadius,
    float seedDistance,
    int numRefinements,
    uint64_t const* bestMatch)
{
    auto direction = Math::unitVectorOfAngle(angle);

    auto distance = seedDistance;
    while (distance <= endRadius) {
        auto scanPos = data.world.getCorrectedPosition(origin + direction * distance);
        if (!data.barrierGrid.hasSolidNearBlockOf(scanPos)) {
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
                return solidDistance;
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
    distance = unpackDistance(bytes);
    density = toFloat((bytes >> 32) & 0xFFFF) / 100;
    angle = convertUint16ToAngle(static_cast<int16_t>((bytes >> 16) & 0xFFFF));
    misc = static_cast<int16_t>(bytes & 0xFFFF);
}

__inline__ __device__ float SensorProcessor::unpackDistance(uint64_t bytes)
{
    return toFloat(bytes >> 48) / DistanceScale;
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
