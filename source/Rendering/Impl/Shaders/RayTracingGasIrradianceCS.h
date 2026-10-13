#pragma once

#include <string_view>

namespace Shaders
{
    // Computes the light of the cells and the light of the radiation sources at the nodes of the irradiance grid. Each light cluster shines
    // from the center of its light, which fades with the distance and is weakened by the gas and the cells in the gas grid on the way, so that
    // dense gas and creatures cast shadows. Zoomed out, the light of the cells spreads over a larger falloff distance near its source, while
    // it stays the same farther away. The radiation sources shine from their areas, and their light passes the exact objects along rays, so
    // that it forms columns of light and shadow. The threads of a work group row share a node and split its light clusters, rays and gas
    // samples, see RayTracingSteps.cpp.
    std::string_view const RayTracingGasIrradianceCS = R"(
const int ShadowSteps = 8;
const float GasExtinction = 0.2;
const float CellExtinction = 1.0;

// Light passing through tissue takes on its color, solids block it
const float TissueExtinction = 6.0;
const float SolidExtinction = 8.0;

// The light of a radiation source is limited, so that it approaches the brightness of the source at its area, while weak light stays
// visible. Each source is limited by itself, so a brighter source never darkens the light of another one. The shadows keep their share of
// the light, so that the columns of light remain at every strength.
const float SourceLightStrength = 4.0;
const float SourceLightKnee = 1.0;

// The light reaches a position from the part of the area of a source next to it. Its half width grows with this factor times the distance
// to the area, so that the shadows become softer the farther they fall.
const float SourceLightSpread = 0.25;

// Wider shining parts trace more rays, so that the beams do not exceed their maximum width at the area, up to this number of rays
const int MaxSourceRays = 16;

// Strands and triangles are sampled at points across a beam
const int BeamSamples = 4;

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

// The gas is only known within the gas grid. The part of the path inside it is sampled about every second node, so that long paths do not
// skip clouds.
float getGasTransmittance(vec2 position, vec2 direction, float pathLength)
{
    vec2 gridEnd = frame.gasGrid.xy + vec2(frame.gasSize.xy - 1) * frame.gasGrid.z;
    float enter = 0.0;
    float exit = pathLength;
    clipSlab(position.x, direction.x, frame.gasGrid.x, gridEnd.x, enter, exit);
    clipSlab(position.y, direction.y, frame.gasGrid.y, gridEnd.y, enter, exit);
    if (exit <= enter) {
        return 1.0;
    }
    int numSteps = clamp(int(ceil((exit - enter) / (GasStepInNodes * frame.gasGrid.z))), 1, MaxGasSteps);
    float stepLength = (exit - enter) / float(numSteps);
    float opticalDepth = 0.0;
    for (int i = int(getNodeThread()); i < numSteps; i += int(NodeThreads)) {
        opticalDepth += sampleGasLayer(position + direction * (enter + stepLength * (float(i) + 0.5)), GasLayer_Clouds).a;
    }
    return exp(-GasExtinction * subgroupClusteredAdd(opticalDepth, NodeThreads) * stepLength);
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

float getStrandChord(vec2 origin, vec2 direction, float pathLength, vec2 a, vec2 b)
{
    vec2 axis = b - a;
    float axisLength = length(axis);
    if (axisLength < 1e-4) {
        return 0.0;
    }
    vec2 along = axis / axisLength;
    vec2 across = vec2(-along.y, along.x);
    float enter = 0.0;
    float exit = pathLength;
    clipSlab(dot(origin - a, along), dot(direction, along), 0.0, axisLength, enter, exit);
    clipSlab(dot(origin - a, across), dot(direction, across), -StrandHalfWidth, StrandHalfWidth, enter, exit);
    return max(exit - enter, 0.0);
}

float getTriangleChord(vec2 origin, vec2 direction, float pathLength, vec2 a, vec2 b, vec2 c)
{
    vec2 corners[3] = vec2[3](a, b, c);
    vec2 ab = b - a;
    vec2 ac = c - a;
    float orientation = ab.x * ac.y - ab.y * ac.x < 0.0 ? -1.0 : 1.0;
    float enter = 0.0;
    float exit = pathLength;
    for (int i = 0; i < 3; ++i) {
        vec2 corner = corners[i];
        vec2 edge = corners[(i + 1) % 3] - corner;
        vec2 inward = orientation * vec2(-edge.y, edge.x);
        clipSlab(dot(origin - corner, inward), dot(direction, inward), 0.0, 1e20, enter, exit);
    }
    return max(exit - enter, 0.0);
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

// Disks at the ends of the path are cut by it, there the chord along the ray counts
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
    return (getDiskChordIntegral(distance + 0.5 * width, radius) - getDiskChordIntegral(distance - 0.5 * width, radius)) / width;
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
    float sum = 0.0;
    for (int i = 0; i < BeamSamples; ++i) {
        float offset = ((float(i) + 0.5) / float(BeamSamples) - 0.5) * width;
        sum += getStrandChord(origin + across * offset, direction, pathLength, a, b);
    }
    return sum / float(BeamSamples);
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
    float sum = 0.0;
    for (int i = 0; i < BeamSamples; ++i) {
        float offset = ((float(i) + 0.5) / float(BeamSamples) - 0.5) * width;
        sum += getTriangleChord(origin + across * offset, direction, pathLength, a, b, c);
    }
    return sum / float(BeamSamples);
}

// The ray stands for a beam with the given widths at its start and its end and collects the average length of its path inside every cell,
// strand and filled triangle on its way. Objects narrower than the beam therefore cast fainter shadows instead of being hit or missed, which
// would cut thin shadows into dashes.
vec3 getObjectTransmittance(vec2 origin, vec2 direction, float pathLength, vec2 beamWidths, vec3 lightColor)
{
    rayQueryEXT query;
    rayQueryInitializeEXT(query, scene, gl_RayFlagsNoneEXT, 0xffu, vec3(origin, 0.5), 0.0, vec3(direction, 0.0), pathLength);
    vec3 maxOpticalDepth = mix(vec3(MaxOpticalDepth), vec3(0.0), equal(lightColor, vec3(0.0)));
    vec3 opticalDepth = vec3(0.0);
    while (rayQueryProceedEXT(query)) {
        uint kind = uint(rayQueryGetIntersectionGeometryIndexEXT(query, false)) + 1u;
        uint primitive = uint(rayQueryGetIntersectionPrimitiveIndexEXT(query, false));
        if (kind == Kind_Sphere) {
            Sphere sphere = spheres[primitive];
            opticalDepth +=
                getAverageDiskChord(origin, direction, pathLength, beamWidths, sphere.centerRadius.xy, sphere.centerRadius.w) * getAbsorption(sphere);
        } else if (kind == Kind_Capsule) {
            uint indexA;
            uint indexB;
            getCapsuleSpheres(primitive, indexA, indexB);
            Sphere a = spheres[indexA];
            Sphere b = spheres[indexB];
            float chord = getAverageStrandChord(origin, direction, pathLength, beamWidths, a.centerRadius.xy, b.centerRadius.xy);
            opticalDepth += chord * 0.5 * (getAbsorption(a) + getAbsorption(b));
        } else if (kind == Kind_Triangle) {
            uint indexA;
            uint indexB;
            uint indexC;
            getTriangleSpheres(primitive, indexA, indexB, indexC);
            Sphere a = spheres[indexA];
            Sphere b = spheres[indexB];
            Sphere c = spheres[indexC];
            float chord = getAverageTriangleChord(origin, direction, pathLength, beamWidths, a.centerRadius.xy, b.centerRadius.xy, c.centerRadius.xy);
            opticalDepth += chord * (getAbsorption(a) + getAbsorption(b) + getAbsorption(c)) / 3.0;
        }
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
// into dashes. Within the area, the light comes from all sides and casts no shadows.
vec3 getSourceTransmittance(SourceLight light, vec2 position, vec2 closestPoint, float distance)
{
    if (distance < 1e-3) {
        return vec3(1.0);
    }
    vec2 direction = (closestPoint - position) / distance;
    vec2 across = vec2(-direction.y, direction.x);
    float nodeWidth = frame.gasLight.z * (abs(across.x) + abs(across.y));
    float areaHalfWidth = isCircular(light) ? light.extent.x : dot(abs(across), light.extent.xy);
    float shiningWidth = 2.0 * min(SourceLightSpread * distance, areaHalfWidth);
    int numRays = clamp(int(ceil(shiningWidth / getMaxBeamWidth())), NodeRayStarts, MaxSourceRays);
    vec2 beamWidths = vec2(nodeWidth, shiningWidth) / float(numRays);
    vec4 weightedTransmittance = vec4(0.0);
    for (int i = int(getNodeThread()); i < numRays; i += int(NodeThreads)) {
        float fraction = (float(i) + 0.5) / float(numRays) - 0.5;
        vec2 target = getClosestSourcePoint(light, closestPoint + across * (fraction * shiningWidth));
        vec2 start = position + across * (fraction * nodeWidth);
        vec2 toTarget = target - start;
        float targetDistance = length(toTarget);
        float closeness = distance / max(targetDistance, distance);
        float weight = closeness * closeness;
        vec3 rayTransmittance = vec3(1.0);
        if (targetDistance > 1e-3) {
            rayTransmittance = getObjectTransmittance(start, toTarget / targetDistance, targetDistance, beamWidths, light.color.rgb);
        }
        weightedTransmittance += vec4(weight * rayTransmittance, weight);
    }
    weightedTransmittance = subgroupClusteredAdd(weightedTransmittance, NodeThreads);
    return weightedTransmittance.rgb / weightedTransmittance.w * getGasTransmittance(position, direction, distance);
}

// The light is strongest at the area of the source and fades smoothly to its range beyond it
vec3 getSourceIrradiance(SourceLight light, vec2 position)
{
    float range = light.centerRangeShape.z;
    vec2 closestPoint = getClosestSourcePoint(light, position);
    float distance = length(closestPoint - position);
    if (distance >= range) {
        return vec3(0.0);
    }
    float fade = 1.0 - distance / range;
    float strength = SourceLightStrength * fade * fade;
    float limitedStrength = strength / (SourceLightKnee + strength);
    return light.color.rgb * limitedStrength * getSourceTransmittance(light, position, closestPoint, distance);
}

// In the periodic world, the copies of a source around the position shine as well. Along an axis that the area of the source spans
// completely, its copies adjoin and form one light, which shines only from the copy next to the position.
vec3 getPeriodicSourceIrradiance(SourceLight light, vec2 position)
{
    vec2 world = frame.world.xy;
    vec2 nearestCenter = getNearestSourceCenter(light, position);
    ivec2 copyRange = ivec2(lessThan(2.0 * light.extent.xy, world));
    vec3 result = vec3(0.0);
    for (int copyY = -copyRange.y; copyY <= copyRange.y; ++copyY) {
        for (int copyX = -copyRange.x; copyX <= copyRange.x; ++copyX) {
            light.centerRangeShape.xy = nearestCenter + world * vec2(copyX, copyY);
            result += getSourceIrradiance(light, position);
        }
    }
    return result;
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
    if (any(greaterThanEqual(node, frame.gasLightGrid.zw))) {
        return;
    }
    vec2 position = frame.gasGrid.xy + vec2(node) * frame.gasLight.z;

    // The light of a cluster comes from the objects within one cluster around it
    ivec2 lowerCluster;
    ivec2 upperCluster;
    getClusterBounds(position, frame.gasLight.y, lowerCluster, upperCluster);
    vec3 irradiance = vec3(0.0);
    uint clusterIndex = 0u;
    for (int y = lowerCluster.y; y <= upperCluster.y; ++y) {
        for (int x = lowerCluster.x; x <= upperCluster.x; ++x) {
            if (clusterIndex++ % NodeThreads == getNodeThread()) {
                irradiance += getCellLight(position, ivec2(x, y));
            }
        }
    }
    irradiance = subgroupClusteredAdd(irradiance, NodeThreads);

    vec3 sourceIrradiance = vec3(0.0);
    for (int i = 0; i < frame.numSourceLights.x; ++i) {
        sourceIrradiance += getPeriodicSourceIrradiance(frame.sourceLights[i], position);
    }
    if (getNodeThread() == 0u) {
        imageStore(gasIrradianceImage, node, vec4(irradiance, 1.0));
        imageStore(gasIrradianceImage, node + ivec2(GasIrradianceImageSize, 0), vec4(sourceIrradiance, 1.0));
    }
}
)";
}
