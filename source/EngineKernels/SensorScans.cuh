#pragma once

#include <cstdint>

#include <Data/CellTypeConstants.h>

#include <cuda_runtime.h>

// Domain decomposition: a sensor scans its rays as long as they stay in the region of its domain. The owners of the strips the
// rays continue into scan the remaining segments in the next time step, and the sensor publishes the combined result in the
// sync round after that, before its signals are read in the next cell function cycle.
struct SensorScan
{
    static auto constexpr NumRays = 64;
    static auto constexpr NoContinuation = -1.0f;
    static auto constexpr NoSegment = -1.0f;
    static auto constexpr NoOcclusion = 1.0e9f;
};

enum class SensorScanType : uint8_t
{
    Rays,
    Relocation,
};

// The properties of a sensor that a scan needs, so that other domains can scan without the sensor
struct SensorScanParams
{
    SensorScanType type;
    SensorMode mode;
    float2 origin;
    float startRadius;
    float endRadius;
    float seedAngle;
    float minDensity;
    uint16_t restrictToColors;
    uint16_t lastMatchCreatureIdPart;
    uint32_t minNumCells;
    uint32_t maxNumCells;
    LineageRestriction restrictToLineage;
    uint32_t lineageId;
    uint64_t creatureId;
    float2 relocationCenter;
};

struct Object;

// A scan whose rays leave the region of the domain, handed over from the sensor kernel to the kernel that requests the continuations
struct SensorContinuation
{
    Object* sensor;
    uint64_t bestMatch;
    SensorScanParams params;
    float starts[SensorScan::NumRays];  // First sample behind the region, NoContinuation for rays with a result in the region
    float ends[SensorScan::NumRays];
};

struct SensorScanRequest
{
    uint8_t requesterDomain;
    uint8_t targetDomain;
    uint32_t pendingIndex;
    uint64_t sensorId;
    uint64_t timestep;
    uint64_t bestMatch;  // Best match of the requester, rays beyond it cannot improve the result
    SensorScanParams params;

    // Segment [start, end) of each ray in the target domain, the starts lie on the sample grid of the ray
    float segmentStarts[SensorScan::NumRays];  // NoSegment for rays without a segment
    float segmentEnds[SensorScan::NumRays];
};

struct SensorScanResponse
{
    uint8_t targetDomain;
    uint32_t pendingIndex;
    uint64_t sensorId;
    uint64_t timestep;
    uint64_t matches[SensorScan::NumRays];          // First match on each segment that no solid on the segment occludes
    float occlusionDistances[SensorScan::NumRays];  // First solid on each segment, NoOcclusion if there is none
};

// A scan waiting for the segments scanned by other domains
struct PendingSensorScan
{
    bool active;
    uint64_t sensorId;
    uint64_t timestep;
    float referenceAngle;  // Direction of the sensor at the time of the scan
    SensorScanParams params;
    uint64_t bestMatch;
    uint64_t matches[SensorScan::NumRays];
    int occlusionDistances[SensorScan::NumRays];  // Float bits, which are ordered like the non-negative floats
};
