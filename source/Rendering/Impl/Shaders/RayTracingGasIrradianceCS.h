#pragma once

#include <string_view>

namespace Shaders
{
    // Computes the light of the cells and the light of the radiation sources at the nodes of the irradiance grid. Each light cluster shines
    // from the center of its light, which fades with the distance and is weakened by the gas and the cells in the gas grid on the way, so that
    // dense gas and creatures cast shadows. Zoomed out, the light of the cells spreads over a larger falloff distance near its source, while
    // it stays the same farther away. The radiation sources shine from their areas, and their light passes the exact objects along rays, so
    // that it forms columns of light and shadow. The threads of a work group row share a node and split its light clusters, rays and gas
    // samples, see SimulationRenderer.cpp.
    std::string_view const RayTracingGasIrradianceCS = R"(
const int ShadowSteps = 8;
const float CellExtinction = 1.0;

// The light of a radiation source is limited, so that it approaches the brightness of the source at its area, while weak light stays
// visible. Each source is limited by itself, so a brighter source never darkens the light of another one. The shadows keep their share of
// the light, so that the columns of light remain at every strength.
const float SourceLightStrength = 4.0;
const float SourceLightKnee = 1.0;

// Wider shining parts trace more rays, so that the beams do not exceed their maximum width at the area, up to this number of rays
const int MaxSourceRays = 16;

// Strands and triangles are sampled at points across a beam, see getBeamSampleOffsets
const int BeamSamples = 4;
const float Pi = 3.14159265;

// A ray stops once the light of the source is blocked in all of its colors
const float MaxOpticalDepth = 5.0;

const float GasStepInNodes = 2.0;
const int MaxGasSteps = 64;

const uint NodeThreads = gl_WorkGroupSize.x;

uint getNodeThread()
{
    return gl_LocalInvocationID.x;
}

float getExtinction(vec2 position)
{
    return GasExtinction * sampleGasLayer(position, GasLayer_Clouds).a + CellExtinction * sampleGasLayer(position, GasLayer_Cells).x;
}

// The path ends one cluster before the light, so that the cells around a light do not shadow it
float getTransmittance(vec2 position, vec2 direction, float distance)
{
    float pathLength = distance - frame.gasLight.x;
    if (pathLength <= 0.0) {
        return 1.0;
    }
    float stepLength = pathLength / float(ShadowSteps);
    float opticalDepth = 0.0;
    for (int i = 0; i < ShadowSteps; ++i) {
        opticalDepth += getExtinction(position + direction * (stepLength * (float(i) + 0.5)));
    }
    return exp(-opticalDepth * stepLength);
}

// Narrows the part of a path inside the slab between two bounds of a coordinate, which changes along the path with the given rate
void clipSlab(float start, float rate, float lower, float upper, inout float enter, inout float exit)
{
    if (abs(rate) < 1e-6) {
        if (start < lower || start > upper) {
            exit = -1.0;
        }
        return;
    }
    float t0 = (lower - start) / rate;
    float t1 = (upper - start) / rate;
    enter = max(enter, min(t0, t1));
    exit = min(exit, max(t0, t1));
}

// Part of a path inside the rectangle from the first to the last node of a grid
bool clipToGrid(vec2 origin, vec2 direction, float pathLength, vec2 firstNode, vec2 lastNode, out float enter, out float exit)
{
    enter = 0.0;
    exit = pathLength;
    clipSlab(origin.x, direction.x, firstNode.x, lastNode.x, enter, exit);
    clipSlab(origin.y, direction.y, firstNode.y, lastNode.y, enter, exit);
    return exit > enter;
}

vec4 sampleShadowGrid(vec2 world)
{
    vec2 texel = (world - frame.shadowGrid.xy) / frame.shadowGrid.z + 0.5;
    return sampleRegionLinear(shadowGridSampler, texel, vec2(0.0), vec2(frame.shadowGridSize.xy), vec2(ShadowGridImageSize));
}

// The objects outside the extracted region are only known by their extinction in the shadow grid, rgb for the tissue and the solids and a for
// the gas, see Shaders/RayTracingShadowGridCS.h. The part of the path inside the grid is sampled about every node, the threads of a node may
// share the samples.
vec4 getShadowGridOpticalDepth(vec2 origin, vec2 direction, float pathLength, int firstSample, int sampleStride)
{
    vec2 lastNode = frame.shadowGrid.xy + vec2(frame.shadowGridSize.xy - 1) * frame.shadowGrid.z;
    float enter;
    float exit;
    if (any(lessThan(frame.shadowGridSize.xy, ivec2(2))) || !clipToGrid(origin, direction, pathLength, frame.shadowGrid.xy, lastNode, enter, exit)) {
        return vec4(0.0);
    }
    int numSamples = max(int(ceil((exit - enter) / frame.shadowGrid.z)), 1);
    float stepLength = (exit - enter) / float(numSamples);
    vec4 extinction = vec4(0.0);
    for (int i = firstSample; i < numSamples; i += sampleStride) {
        extinction += sampleShadowGrid(origin + direction * (enter + stepLength * (float(i) + 0.5)));
    }
    return extinction * stepLength;
}

// The gas is known within the gas grid, which is sampled about every second node, so that long paths do not skip clouds, and outside the
// extracted region in the shadow grid
float getGasTransmittance(vec2 position, vec2 direction, float pathLength)
{
    float opticalDepth = getShadowGridOpticalDepth(position, direction, pathLength, int(getNodeThread()), int(NodeThreads)).a;
    vec2 lastNode = frame.gasGrid.xy + vec2(frame.gasSize.xy - 1) * frame.gasGrid.z;
    float enter;
    float exit;
    if (clipToGrid(position, direction, pathLength, frame.gasGrid.xy, lastNode, enter, exit)) {
        int numSteps = clamp(int(ceil((exit - enter) / (GasStepInNodes * frame.gasGrid.z))), 1, MaxGasSteps);
        float stepLength = (exit - enter) / float(numSteps);
        float density = 0.0;
        for (int i = int(getNodeThread()); i < numSteps; i += int(NodeThreads)) {
            density += sampleGasLayer(position + direction * (enter + stepLength * (float(i) + 0.5)), GasLayer_Clouds).a;
        }
        opticalDepth += GasExtinction * density * stepLength;
    }
    return exp(-subgroupClusteredAdd(opticalDepth, NodeThreads));
}

float getDiskChord(vec2 origin, vec2 direction, float pathLength, vec2 center, float radius)
{
    vec2 toCenter = center - origin;
    float along = dot(toCenter, direction);
    float squaredDistance = dot(toCenter, toCenter) - along * along;
    if (squaredDistance >= radius * radius) {
        return 0.0;
    }
    float halfChord = sqrt(radius * radius - squaredDistance);
    return max(min(along + halfChord, pathLength) - max(along - halfChord, 0.0), 0.0);
}

// The samples across a beam start their paths at the given offsets across it, see getAverageStrandChord. The coordinate of the start of a
// path changes linearly with its offset, while all paths change the coordinate at the same rate. Narrows the parts of the paths inside the
// slab, see clipSlab.
void clipSlabSamples(float start, float shift, float rate, float lower, float upper, vec4 offsets, inout vec4 enter, inout vec4 exit)
{
    vec4 starts = start + offsets * shift;
    if (abs(rate) < 1e-6) {
        exit = mix(exit, vec4(-1.0), notEqual(clamp(starts, vec4(lower), vec4(upper)), starts));
        return;
    }
    float inverseRate = 1.0 / rate;
    vec4 t0 = (lower - starts) * inverseRate;
    vec4 t1 = (upper - starts) * inverseRate;
    enter = max(enter, min(t0, t1));
    exit = min(exit, max(t0, t1));
}

// The side of a triangle from the corner to the next one bounds the slab towards the inside of the triangle
void clipTriangleSideSamples(
    vec2 origin, vec2 direction, vec2 across, vec2 corner, vec2 nextCorner, float orientation, vec4 offsets, inout vec4 enter, inout vec4 exit)
{
    vec2 side = nextCorner - corner;
    vec2 inward = orientation * vec2(-side.y, side.x);
    clipSlabSamples(dot(origin - corner, inward), dot(across, inward), dot(direction, inward), 0.0, 1e20, offsets, enter, exit);
}

vec3 getAbsorption(Sphere sphere)
{
    if (sphere.hueMaterial.w == Material_Solid) {
        return vec3(SolidExtinction);
    }
    return TissueExtinction * getAbsorbedShare(getSaturatedHue(sphere.hueMaterial.rgb));
}

// Integral of the chord lengths of a disk around the origin over the distances of the chords from the center up to the given one. The
// clamping keeps rounding errors at the edge of the disk away from the square root and the arc sine.
float getDiskChordIntegral(float distance, float radius)
{
    float x = clamp(distance, -radius, radius);
    return x * sqrt(max(radius * radius - x * x, 0.0)) + radius * radius * asin(clamp(x / radius, -1.0, 1.0));
}

// A beam has the given widths at the start and at the end of the path and changes its width linearly in between. It is not wider than the
// enlargement of the bounding boxes allows.
float getBeamWidth(vec2 beamWidths, float along, float pathLength)
{
    return min(mix(beamWidths.x, beamWidths.y, clamp(along / pathLength, 0.0, 1.0)), getMaxBeamWidth());
}

// Disks at the ends of the path are cut by it, there the chord along the ray counts. The integral is odd in the distance and reaches half
// the area of the disk at its edge, so that a beam covering the whole disk needs no arc sine.
float getAverageDiskChord(vec2 origin, vec2 direction, float pathLength, vec2 beamWidths, vec2 center, float radius)
{
    vec2 toCenter = center - origin;
    float along = dot(toCenter, direction);
    float width = getBeamWidth(beamWidths, along, pathLength);
    float distance = dot(toCenter, vec2(-direction.y, direction.x));
    if (abs(distance) >= radius + 0.5 * width) {
        return 0.0;
    }
    if (along < radius || along > pathLength - radius || width < 1e-3) {
        return getDiskChord(origin, direction, pathLength, center, radius);
    }
    float halfArea = 0.5 * Pi * radius * radius;
    float outerEdge = abs(distance) + 0.5 * width;
    float innerEdge = 0.5 * width - abs(distance);
    float outer = outerEdge < radius ? getDiskChordIntegral(outerEdge, radius) : halfArea;
    float inner = innerEdge < radius ? getDiskChordIntegral(innerEdge, radius) : halfArea;
    return (outer + inner) / width;
}

vec4 getBeamSampleOffsets(float width)
{
    return ((vec4(0.0, 1.0, 2.0, 3.0) + 0.5) / float(BeamSamples) - 0.5) * width;
}

// Strands and triangles are sampled at points across the beam where it passes their middle
float getAverageStrandChord(vec2 origin, vec2 direction, float pathLength, vec2 beamWidths, vec2 a, vec2 b)
{
    vec2 across = vec2(-direction.y, direction.x);
    float width = getBeamWidth(beamWidths, dot(0.5 * (a + b) - origin, direction), pathLength);
    vec2 distances = vec2(dot(a - origin, across), dot(b - origin, across));
    float reach = 0.5 * width + StrandHalfWidth;
    if (min(distances.x, distances.y) >= reach || max(distances.x, distances.y) <= -reach) {
        return 0.0;
    }
    vec2 axis = b - a;
    float axisLength = length(axis);
    if (axisLength < 1e-4) {
        return 0.0;
    }
    vec2 along = axis / axisLength;
    vec2 strandAcross = vec2(-along.y, along.x);
    vec2 toOrigin = origin - a;
    vec4 offsets = getBeamSampleOffsets(width);
    vec4 enter = vec4(0.0);
    vec4 exit = vec4(pathLength);
    clipSlabSamples(dot(toOrigin, along), dot(across, along), dot(direction, along), 0.0, axisLength, offsets, enter, exit);
    clipSlabSamples(
        dot(toOrigin, strandAcross), dot(across, strandAcross), dot(direction, strandAcross), -StrandHalfWidth, StrandHalfWidth, offsets, enter, exit);
    vec4 chords = max(exit - enter, vec4(0.0));
    return (chords.x + chords.y + chords.z + chords.w) / float(BeamSamples);
}

float getAverageTriangleChord(vec2 origin, vec2 direction, float pathLength, vec2 beamWidths, vec2 a, vec2 b, vec2 c)
{
    vec2 across = vec2(-direction.y, direction.x);
    float width = getBeamWidth(beamWidths, dot((a + b + c) / 3.0 - origin, direction), pathLength);
    vec3 distances = vec3(dot(a - origin, across), dot(b - origin, across), dot(c - origin, across));
    float reach = 0.5 * width;
    if (minComponent(distances) >= reach || maxComponent(distances) <= -reach) {
        return 0.0;
    }
    vec2 ab = b - a;
    vec2 ac = c - a;
    float orientation = ab.x * ac.y - ab.y * ac.x < 0.0 ? -1.0 : 1.0;
    vec4 offsets = getBeamSampleOffsets(width);
    vec4 enter = vec4(0.0);
    vec4 exit = vec4(pathLength);
    clipTriangleSideSamples(origin, direction, across, a, b, orientation, offsets, enter, exit);
    clipTriangleSideSamples(origin, direction, across, b, c, orientation, offsets, enter, exit);
    clipTriangleSideSamples(origin, direction, across, c, a, orientation, offsets, enter, exit);
    vec4 chords = max(exit - enter, vec4(0.0));
    return (chords.x + chords.y + chords.z + chords.w) / float(BeamSamples);
}

// Optical depth that an object found by the ray query adds along the beam, see getObjectTransmittance
vec3 getCandidateOpticalDepth(uint kind, uint primitive, vec2 origin, vec2 direction, float pathLength, vec2 beamWidths)
{
    if (kind == Kind_Sphere) {
        Sphere sphere = spheres[primitive];
        return getAverageDiskChord(origin, direction, pathLength, beamWidths, sphere.centerRadius.xy, sphere.centerRadius.w) * getAbsorption(sphere);
    }
    if (kind == Kind_Capsule) {
        uint indexA;
        uint indexB;
        getCapsuleSpheres(primitive, indexA, indexB);
        Sphere a = spheres[indexA];
        Sphere b = spheres[indexB];
        float chord = getAverageStrandChord(origin, direction, pathLength, beamWidths, a.centerRadius.xy, b.centerRadius.xy);
        return chord * 0.5 * (getAbsorption(a) + getAbsorption(b));
    }
    if (kind == Kind_Triangle) {
        uint indexA;
        uint indexB;
        uint indexC;
        getTriangleSpheres(primitive, indexA, indexB, indexC);
        Sphere a = spheres[indexA];
        Sphere b = spheres[indexB];
        Sphere c = spheres[indexC];
        float chord = getAverageTriangleChord(origin, direction, pathLength, beamWidths, a.centerRadius.xy, b.centerRadius.xy, c.centerRadius.xy);
        return chord * (getAbsorption(a) + getAbsorption(b) + getAbsorption(c)) / 3.0;
    }
    return vec3(0.0);
}

// The ray stands for a beam with the given widths at its start and its end and collects the average length of its path inside every cell,
// strand and filled triangle on its way. Objects narrower than the beam therefore cast fainter shadows instead of being hit or missed, which
// would cut thin shadows into dashes. The objects outside the extracted region add their extinction in the shadow grid.
vec3 getObjectTransmittance(vec2 origin, vec2 direction, float pathLength, vec2 beamWidths, vec3 lightColor)
{
    vec3 maxOpticalDepth = mix(vec3(MaxOpticalDepth), vec3(0.0), equal(lightColor, vec3(0.0)));
    vec3 opticalDepth = getShadowGridOpticalDepth(origin, direction, pathLength, 0, 1).rgb;
    if (all(greaterThanEqual(opticalDepth, maxOpticalDepth))) {
        return exp(-opticalDepth);
    }
    rayQueryEXT query;
    rayQueryInitializeEXT(query, scene, gl_RayFlagsNoneEXT, 0xffu, vec3(origin, 0.5), 0.0, vec3(direction, 0.0), pathLength);
    while (rayQueryProceedEXT(query)) {
        uint kind = uint(rayQueryGetIntersectionGeometryIndexEXT(query, false)) + 1u;
        uint primitive = uint(rayQueryGetIntersectionPrimitiveIndexEXT(query, false));
        opticalDepth += getCandidateOpticalDepth(kind, primitive, origin, direction, pathLength, beamWidths);
        if (all(greaterThanEqual(opticalDepth, maxOpticalDepth))) {
            rayQueryTerminateEXT(query);
        }
    }
    return exp(-opticalDepth);
}
)"
                                                       R"(
struct ClusterLight
{
    vec3 color;
    vec2 toLight;
    float distance;
};

ClusterLight getClusterLight(vec2 position, ivec2 cluster)
{
    ClusterLight result = ClusterLight(vec3(0.0), vec2(0.0), 0.0);
    uint entry = uint(cluster.y * frame.gasLightGrid.x + cluster.x) * GasLightClusterStride;
    float lightLuminance = float(gasLightClusters[entry + 2u]);
    if (lightLuminance > 0.0) {
        vec2 lightOffset = vec2(float(gasLightClusters[entry]), float(gasLightClusters[entry + 1u])) / lightLuminance - GasLightOffsetShift;
        result.toLight = frame.gasLighting.xy + (vec2(cluster) + lightOffset) * frame.gasLight.x - position;
        result.distance = length(result.toLight);
        result.color = vec3(float(gasLightClusters[entry + 3u]), float(gasLightClusters[entry + 4u]), float(gasLightClusters[entry + 5u])) / GasFixedPointScale;
    }
    return result;
}

vec3 getCellLight(vec2 position, ivec2 cluster)
{
    float range = frame.gasLight.y;
    float falloffDistance = frame.gasLight.w;
    ClusterLight light = getClusterLight(position, cluster);
    if (light.distance >= range || luminance(light.color) <= 0.0) {
        return vec3(0.0);
    }
    float falloff = LightFalloffDistance * LightFalloffDistance / (falloffDistance * falloffDistance + light.distance * light.distance)
        * (1.0 - smoothstep(range * 0.6, range, light.distance));
    return light.color * falloff * getTransmittance(position, light.toLight / max(light.distance, 1e-4), light.distance);
}

// The light comes from the shining part of the area next to the position, which spreads across the direction to the area. The rays run from
// points across the width of the node to points across the shining part, closer points count more. They stand for beams that lead from
// the parts of the node to the parts of the shining part side by side, so that thin shadows are averaged over the node instead of being cut
// into dashes. Within the area, the light comes from all sides and casts no shadows. MaxSourceRays does not exceed NodeThreads, so that each
// thread of a node traces at most one ray of a source.
struct SourceBeams
{
    vec2 direction;
    vec2 across;
    float nodeWidth;
    float shiningWidth;
    int numRays;
    vec2 beamWidths;
};

SourceBeams getSourceBeams(SourceLight light, vec2 position, vec2 closestPoint, float distance)
{
    vec2 direction = (closestPoint - position) / distance;
    vec2 across = vec2(-direction.y, direction.x);
    float nodeWidth = frame.gasLight.z * (abs(across.x) + abs(across.y));
    float areaHalfWidth = isCircular(light) ? light.extent.x : dot(abs(across), light.extent.xy);
    float shiningWidth = 2.0 * min(SourceLightSpread * distance, areaHalfWidth);
    int numRays = clamp(int(ceil(shiningWidth / getMaxBeamWidth())), NodeRayStarts, MaxSourceRays);
    return SourceBeams(direction, across, nodeWidth, shiningWidth, numRays, vec2(nodeWidth, shiningWidth) / float(numRays));
}

struct SourceRay
{
    vec2 start;
    vec2 direction;
    float pathLength;
    float weight;
};

SourceRay getSourceRay(SourceLight light, vec2 position, vec2 closestPoint, float distance, SourceBeams beams, int index)
{
    float fraction = (float(index) + 0.5) / float(beams.numRays) - 0.5;
    vec2 target = getClosestSourcePoint(light, closestPoint + beams.across * (fraction * beams.shiningWidth));
    vec2 start = position + beams.across * (fraction * beams.nodeWidth);
    vec2 toTarget = target - start;
    float targetDistance = length(toTarget);
    float closeness = distance / max(targetDistance, distance);
    return SourceRay(start, toTarget / targetDistance, targetDistance, closeness * closeness);
}

// The threads of the node trace the rays of a source together, for the sources beyond the batch of the node, see main
vec3 getSourceTransmittance(SourceLight light, vec2 position, vec2 closestPoint, float distance)
{
    SourceBeams beams = getSourceBeams(light, position, closestPoint, distance);
    int index = int(getNodeThread());
    vec4 weightedTransmittance = vec4(0.0);
    if (index < beams.numRays) {
        SourceRay ray = getSourceRay(light, position, closestPoint, distance, beams, index);
        vec3 rayTransmittance = vec3(1.0);
        if (ray.pathLength > 1e-3) {
            rayTransmittance = getObjectTransmittance(ray.start, ray.direction, ray.pathLength, beams.beamWidths, light.color.rgb);
        }
        weightedTransmittance = vec4(ray.weight * rayTransmittance, ray.weight);
    }
    weightedTransmittance = subgroupClusteredAdd(weightedTransmittance, NodeThreads);
    return weightedTransmittance.rgb / weightedTransmittance.w * getGasTransmittance(position, beams.direction, distance);
}
)"
                                                       R"(
// The rays of the sources differ greatly in their cost, so the threads of a node would wait for the slowest ray of each source. Hence the
// nodes of a work group first collect a batch of their sources in lockstep, then all threads of the work group trace the rays of the batches,
// each taking the next ray as soon as its previous one is finished, and finally each node sums the light of its batch in a fixed order. A
// batch holds up to MaxBatchSources sources, which does not exceed NodeThreads. Each ray keeps a value: the optical depth of the shadow
// grid, which is sampled in lockstep, and then its weighted transmittance.
const uint NodesPerGroup = gl_WorkGroupSize.y;
const uint MaxBatchSources = 8u;
const uint RaysPerNode = MaxBatchSources * uint(MaxSourceRays);

struct BatchSource
{
    vec4 centerClosestPoint;  // xy: center of the copy of the source, zw: closest point of its area to the node
    vec4 extentShape;         // xy: extent, z: shape, w: distance of the closest point
    vec3 lightPerWeight;      // Light of the source per weight of its rays
    uint numRays;
};

shared BatchSource batchSources[NodesPerGroup][MaxBatchSources];
shared uint numBatchSources[NodesPerGroup];
shared float rayValues[3u * NodesPerGroup * RaysPerNode];
shared uint nextRay;

uint getRaySlot(uint nodeInGroup, uint source, uint index)
{
    return (nodeInGroup * MaxBatchSources + source) * uint(MaxSourceRays) + index;
}

vec3 getRayValue(uint slot)
{
    return vec3(rayValues[3u * slot], rayValues[3u * slot + 1u], rayValues[3u * slot + 2u]);
}

void setRayValue(uint slot, vec3 value)
{
    rayValues[3u * slot] = value.x;
    rayValues[3u * slot + 1u] = value.y;
    rayValues[3u * slot + 2u] = value.z;
}

vec2 getNodePosition(ivec2 node)
{
    return frame.gasGrid.xy + vec2(node) * frame.gasLight.z;
}

// The pixels sample the nodes within a few nodes around them. Borderless rendering repeats the part of the view of the size of the world, see
// Shaders/ModuloCopyFS.h. Without borderless rendering, the space outside the world only shows the gas next to its border.
bool isSampled(ivec2 node)
{
    if (node.y >= frame.gasLightGrid.w) {
        return false;
    }
    vec2 lower = frame.rect.xy;
    vec2 upper = frame.rect.xy + frame.view.xy / frame.view.z;
    if (frame.world.z > 0.5) {
        upper = min(upper, lower + frame.world.xy);
    } else {
        float gasReach = float(frame.gasSize.z + 3) * frame.gasGrid.z;
        lower = max(lower, vec2(-gasReach));
        upper = min(upper, frame.world.xy + gasReach);
    }
    float samplingReach = 3.0 * frame.gasLight.z + 1.0 / frame.view.z;
    vec2 position = getNodePosition(node);
    return all(greaterThanEqual(position, lower - samplingReach)) && all(lessThanEqual(position, upper + samplingReach));
}

void addBatchSource(uint nodeInGroup, uint source, SourceLight light, vec2 position, vec2 closestPoint, float distance, vec3 sourceLight)
{
    SourceBeams beams = getSourceBeams(light, position, closestPoint, distance);
    int index = int(getNodeThread());
    float weight = 0.0;
    if (index < beams.numRays) {
        SourceRay ray = getSourceRay(light, position, closestPoint, distance, beams, index);
        weight = ray.weight;
        vec3 opticalDepth = ray.pathLength > 1e-3 ? getShadowGridOpticalDepth(ray.start, ray.direction, ray.pathLength, 0, 1).rgb : vec3(0.0);
        setRayValue(getRaySlot(nodeInGroup, source, uint(index)), opticalDepth);
    }
    float weightSum = subgroupClusteredAdd(weight, NodeThreads);
    float gasTransmittance = getGasTransmittance(position, beams.direction, distance);
    if (getNodeThread() == 0u) {
        batchSources[nodeInGroup][source] = BatchSource(
            vec4(light.centerRangeShape.xy, closestPoint),
            vec4(light.extent.xy, light.centerRangeShape.w, distance),
            sourceLight * gasTransmittance / weightSum,
            uint(beams.numRays));
    }
}

// The light is strongest at the area of a source and fades smoothly to its range beyond it. In the periodic world, the copies of a source
// around the position shine as well. Along an axis that the area of the source spans completely, its copies adjoin and form one light,
// which shines only from the copy next to the position. Returns the light that does not depend on the rays of the batch.
vec3 collectBatchSources(uint nodeInGroup, vec2 position)
{
    vec3 result = vec3(0.0);
    uint numSources = 0u;
    vec2 world = frame.world.xy;
    for (int i = 0; i < frame.numSourceLights.x; ++i) {
        SourceLight light = frame.sourceLights[i];
        float range = light.centerRangeShape.z;
        vec2 nearestCenter = getNearestSourceCenter(light, position);
        ivec2 copyRange = ivec2(lessThan(2.0 * light.extent.xy, world));
        for (int copyY = -copyRange.y; copyY <= copyRange.y; ++copyY) {
            for (int copyX = -copyRange.x; copyX <= copyRange.x; ++copyX) {
                light.centerRangeShape.xy = nearestCenter + world * vec2(copyX, copyY);
                vec2 closestPoint = getClosestSourcePoint(light, position);
                float distance = length(closestPoint - position);
                if (distance >= range) {
                    continue;
                }
                float fade = 1.0 - distance / range;
                float strength = SourceLightStrength * fade * fade;
                float limitedStrength = strength / (SourceLightKnee + strength);
                vec3 sourceLight = light.color.rgb * limitedStrength;
                if (distance < 1e-3) {
                    result += sourceLight;
                } else if (numSources == MaxBatchSources) {
                    result += sourceLight * getSourceTransmittance(light, position, closestPoint, distance);
                } else {
                    addBatchSource(nodeInGroup, numSources++, light, position, closestPoint, distance, sourceLight);
                }
            }
        }
    }
    if (getNodeThread() == 0u) {
        numBatchSources[nodeInGroup] = numSources;
    }
    return result;
}

void traceBatchRays()
{
    rayQueryEXT query;
    bool tracing = false;
    uint slot = 0u;
    SourceRay ray;
    vec2 beamWidths;
    vec3 opticalDepth;
    vec3 maxOpticalDepth;
    for (;;) {
        if (!tracing) {
            slot = atomicAdd(nextRay, 1u);
            if (slot >= NodesPerGroup * RaysPerNode) {
                break;
            }
            uint nodeInGroup = slot / RaysPerNode;
            uint source = slot / uint(MaxSourceRays) % MaxBatchSources;
            uint index = slot % uint(MaxSourceRays);
            if (source >= numBatchSources[nodeInGroup]) {
                continue;
            }
            BatchSource batchSource = batchSources[nodeInGroup][source];
            if (index >= batchSource.numRays) {
                continue;
            }
            SourceLight light =
                SourceLight(vec4(batchSource.centerClosestPoint.xy, 0.0, batchSource.extentShape.z), vec4(0.0), vec4(batchSource.extentShape.xy, 0.0, 0.0));
            vec2 position = getNodePosition(ivec2(gl_WorkGroupID.x, gl_WorkGroupID.y * NodesPerGroup + nodeInGroup));
            vec2 closestPoint = batchSource.centerClosestPoint.zw;
            float distance = batchSource.extentShape.w;
            SourceBeams beams = getSourceBeams(light, position, closestPoint, distance);
            ray = getSourceRay(light, position, closestPoint, distance, beams, int(index));
            beamWidths = beams.beamWidths;
            opticalDepth = getRayValue(slot);

            // A ray stops once the light is blocked in all of its colors, the light per weight vanishes in the same colors as the light
            maxOpticalDepth = mix(vec3(MaxOpticalDepth), vec3(0.0), equal(batchSource.lightPerWeight, vec3(0.0)));
            if (ray.pathLength <= 1e-3) {
                setRayValue(slot, vec3(ray.weight));
                continue;
            }
            if (all(greaterThanEqual(opticalDepth, maxOpticalDepth))) {
                setRayValue(slot, ray.weight * exp(-opticalDepth));
                continue;
            }
            rayQueryInitializeEXT(query, scene, gl_RayFlagsNoneEXT, 0xffu, vec3(ray.start, 0.5), 0.0, vec3(ray.direction, 0.0), ray.pathLength);
            tracing = true;
        }
        if (rayQueryProceedEXT(query)) {
            uint kind = uint(rayQueryGetIntersectionGeometryIndexEXT(query, false)) + 1u;
            uint primitive = uint(rayQueryGetIntersectionPrimitiveIndexEXT(query, false));
            opticalDepth += getCandidateOpticalDepth(kind, primitive, ray.start, ray.direction, ray.pathLength, beamWidths);
            if (!all(greaterThanEqual(opticalDepth, maxOpticalDepth))) {
                continue;
            }
            rayQueryTerminateEXT(query);
        }
        setRayValue(slot, ray.weight * exp(-opticalDepth));
        tracing = false;
    }
}

// Each source of the batch is summed by one thread of the node
vec3 getBatchLight(uint nodeInGroup)
{
    uint source = getNodeThread();
    if (source >= numBatchSources[nodeInGroup]) {
        return vec3(0.0);
    }
    BatchSource batchSource = batchSources[nodeInGroup][source];
    vec3 weightedTransmittance = vec3(0.0);
    for (uint index = 0u; index < batchSource.numRays; ++index) {
        weightedTransmittance += getRayValue(getRaySlot(nodeInGroup, source, index));
    }
    return batchSource.lightPerWeight * weightedTransmittance;
}

// The bounds of the loops stay inside the grid, because loops that skipped the clusters outside returned no light at all on an NVIDIA GPU
// with driver 610.88 once the range grew
void getClusterBounds(vec2 position, float range, out ivec2 lowerCluster, out ivec2 upperCluster)
{
    ivec2 center = ivec2(floor((position - frame.gasLighting.xy) / frame.gasLight.x));
    int clusterRange = int(ceil(range / frame.gasLight.x)) + 1;
    lowerCluster = max(center - clusterRange, ivec2(0));
    upperCluster = min(center + clusterRange, frame.gasLightGrid.xy - 1);
}

void main()
{
    ivec2 node = ivec2(gl_WorkGroupID.x, gl_GlobalInvocationID.y);
    uint nodeInGroup = gl_LocalInvocationID.y;
    if (node.x >= frame.gasLightGrid.z) {
        return;
    }
    bool sampled = isSampled(node);
    if (!sampled && node.y < frame.gasLightGrid.w && getNodeThread() == 0u) {
        imageStore(gasIrradianceImage, node, vec4(0.0));
        imageStore(gasIrradianceImage, node + ivec2(GasIrradianceImageSize, 0), vec4(0.0));
    }

    // The barriers require that work groups return as a whole, the nodes of a work group that are not sampled trace rays of the others
    bool groupSampled = false;
    for (uint other = 0u; other < NodesPerGroup; ++other) {
        groupSampled = groupSampled || isSampled(ivec2(node.x, gl_WorkGroupID.y * NodesPerGroup + other));
    }
    if (!groupSampled) {
        return;
    }
    vec2 position = getNodePosition(node);

    // The light of a cluster comes from the objects within one cluster around it
    vec3 irradiance = vec3(0.0);
    vec3 sourceIrradiance = vec3(0.0);
    if (sampled) {
        ivec2 lowerCluster;
        ivec2 upperCluster;
        getClusterBounds(position, frame.gasLight.y, lowerCluster, upperCluster);
        ivec2 numClusters = upperCluster - lowerCluster + 1;
        int clusterCount = all(greaterThan(numClusters, ivec2(0))) ? numClusters.x * numClusters.y : 0;
        for (int index = int(getNodeThread()); index < clusterCount; index += int(NodeThreads)) {
            irradiance += getCellLight(position, lowerCluster + ivec2(index % numClusters.x, index / numClusters.x));
        }
        irradiance = subgroupClusteredAdd(irradiance, NodeThreads);
        sourceIrradiance = collectBatchSources(nodeInGroup, position);
    } else if (getNodeThread() == 0u) {
        numBatchSources[nodeInGroup] = 0u;
    }
    if (gl_LocalInvocationIndex == 0u) {
        nextRay = 0u;
    }
    memoryBarrierShared();
    barrier();
    traceBatchRays();
    memoryBarrierShared();
    barrier();
    if (sampled) {
        sourceIrradiance += subgroupClusteredAdd(getBatchLight(nodeInGroup), NodeThreads);
        if (getNodeThread() == 0u) {
            imageStore(gasIrradianceImage, node, vec4(irradiance, 1.0));
            imageStore(gasIrradianceImage, node + ivec2(GasIrradianceImageSize, 0), vec4(sourceIrradiance, 1.0));
        }
    }
}
)";
}
