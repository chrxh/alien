#pragma once

#include <string_view>

namespace Shaders
{
    // Computes the light of the cells and the light of the radiation sources at the nodes of the irradiance grid. Each light cluster shines
    // from the center of its light, which fades with the distance and is weakened by the gas and the cells in the gas grid on the way, so that
    // dense gas and creatures cast shadows. Zoomed out, the light of the cells spreads over a larger falloff distance near its source, while
    // it stays the same farther away. The radiation sources shine from their areas, and their light passes the exact objects along rays, so
    // that it forms columns of light and shadow.
    std::string_view const RayTracingGasIrradianceCS = R"(
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

// Falloff distance when zoomed in, which must match LightFalloffDistance in Rendering/Impl/RayTracingResources.cpp
const float LightFalloffDistance = 5.0;
const int ShadowSteps = 8;
const float GasExtinction = 0.2;
const float CellExtinction = 1.0;

// Light passing through tissue takes on its color, solids block it
const float TissueExtinction = 6.0;
const float SolidExtinction = 8.0;

// Strength of the light of a radiation source at its area, where the light approaches the brightness of the source
const float SourceLightStrength = 4.0;

// The rays to a source start at least from this many positions within the area of a node
const int NodeRayStarts = 4;

// A ray stops once the light is blocked
const float MaxOpticalDepth = 5.0;

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

float getGasTransmittance(vec2 position, vec2 direction, float pathLength)
{
    float stepLength = pathLength / float(ShadowSteps);
    float opticalDepth = 0.0;
    for (int i = 0; i < ShadowSteps; ++i) {
        opticalDepth += GasExtinction * sampleGasLayer(position + direction * (stepLength * (float(i) + 0.5)), GasLayer_Clouds).a;
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
    vec3 hue = sphere.hueMaterial.rgb;
    return TissueExtinction * (vec3(1.0) - 0.85 * mix(vec3(luminance(hue)), hue, 0.9));
}

// The ray collects the length of its path inside every cell, strand and filled triangle on its way
vec3 getObjectTransmittance(vec2 origin, vec2 direction, float pathLength)
{
    rayQueryEXT query;
    rayQueryInitializeEXT(query, scene, gl_RayFlagsNoneEXT, 0xffu, vec3(origin, 0.5), 0.0, vec3(direction, 0.0), pathLength);
    vec3 opticalDepth = vec3(0.0);
    while (rayQueryProceedEXT(query)) {
        uint kind = uint(rayQueryGetIntersectionGeometryIndexEXT(query, false)) + 1u;
        uint primitive = uint(rayQueryGetIntersectionPrimitiveIndexEXT(query, false));
        if (kind == Kind_Sphere) {
            Sphere sphere = spheres[primitive];
            opticalDepth += getDiskChord(origin, direction, pathLength, sphere.centerRadius.xy, sphere.centerRadius.w) * getAbsorption(sphere);
        } else if (kind == Kind_Capsule) {
            uint indexA;
            uint indexB;
            getCapsuleSpheres(primitive, indexA, indexB);
            Sphere a = spheres[indexA];
            Sphere b = spheres[indexB];
            float chord = getStrandChord(origin, direction, pathLength, a.centerRadius.xy, b.centerRadius.xy);
            opticalDepth += chord * 0.5 * (getAbsorption(a) + getAbsorption(b));
        } else if (kind == Kind_Triangle) {
            uint indexA;
            uint indexB;
            uint indexC;
            getTriangleSpheres(primitive, indexA, indexB, indexC);
            Sphere a = spheres[indexA];
            Sphere b = spheres[indexB];
            Sphere c = spheres[indexC];
            float chord = getTriangleChord(origin, direction, pathLength, a.centerRadius.xy, b.centerRadius.xy, c.centerRadius.xy);
            opticalDepth += chord * (getAbsorption(a) + getAbsorption(b) + getAbsorption(c)) / 3.0;
        }
        if (minComponent(opticalDepth) > MaxOpticalDepth) {
            rayQueryTerminateEXT(query);
        }
    }
    return exp(-opticalDepth);
}

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

bool isCircular(SourceLight light)
{
    return light.colorShape.w < 0.5;
}

float getSourceDistance(SourceLight light, vec2 position)
{
    vec2 offset = position - light.centerRange.xy;
    if (isCircular(light)) {
        return max(length(offset) - light.extent.x, 0.0);
    }
    return length(max(abs(offset) - light.extent.xy, vec2(0.0)));
}

// Length of the line through a rectangle around the origin, which runs along the direction at the offset across it, and the position of
// the middle of the line along the direction
float getRectangleChord(vec2 halfSize, vec2 direction, vec2 across, float offset, out float middle)
{
    vec2 start = across * offset;
    float enter = -1e20;
    float exit = 1e20;
    middle = 0.0;
    for (int axis = 0; axis < 2; ++axis) {
        if (abs(direction[axis]) < 1e-6) {
            if (abs(start[axis]) > halfSize[axis]) {
                return 0.0;
            }
            continue;
        }
        float t0 = (-halfSize[axis] - start[axis]) / direction[axis];
        float t1 = (halfSize[axis] - start[axis]) / direction[axis];
        enter = max(enter, min(t0, t1));
        exit = min(exit, max(t0, t1));
    }
    middle = 0.5 * (enter + exit);
    return max(exit - enter, 0.0);
}

// The area of a source is divided into strips along the direction to the source. A ray runs to the middle of each strip and counts with
// the length of the strip, so that the shadows become soft behind large areas. The rays start at fixed positions within the area of the
// node, so that sharp edges of the shadows are smoothed instead of being cut into steps by the grid. The strips follow the direction, so
// the shadows do not flicker.
vec3 getSourceTransmittance(SourceLight light, vec2 position)
{
    vec2 center = light.centerRange.xy;
    vec2 toCenter = center - position;
    float centerDistance = length(toCenter);
    vec2 direction = centerDistance > 1e-4 ? toCenter / centerDistance : vec2(1.0, 0.0);
    vec2 across = vec2(-direction.y, direction.x);
    bool circular = isCircular(light);
    float halfWidth = circular ? light.extent.x : dot(abs(across), light.extent.xy);
    int numRays = max(int(light.centerRange.w), NodeRayStarts);
    vec3 transmittance = vec3(0.0);
    float weightSum = 0.0;
    for (int i = 0; i < numRays; ++i) {
        float offset = ((float(i) + 0.5) / float(numRays) * 2.0 - 1.0) * halfWidth;
        float middle = 0.0;
        float weight = circular ? sqrt(max(halfWidth * halfWidth - offset * offset, 0.0))
                                : getRectangleChord(light.extent.xy, direction, across, offset, middle);
        weight = max(weight, 1e-6);
        vec2 start = position + (vec2(float(i & 1), float((i >> 1) & 1)) - 0.5) * 0.5 * frame.gasLight.z;
        vec2 toTarget = center + across * offset + direction * middle - start;
        float targetDistance = length(toTarget);
        vec3 rayTransmittance = vec3(1.0);
        if (targetDistance > 1e-3) {
            rayTransmittance = getObjectTransmittance(start, toTarget / targetDistance, targetDistance);
        }
        transmittance += weight * rayTransmittance;
        weightSum += weight;
    }
    float gasTransmittance = centerDistance > 1e-3 ? getGasTransmittance(position, direction, centerDistance) : 1.0;
    return transmittance / max(weightSum, 1e-6) * gasTransmittance;
}

// The shadowed light in the color of the source and the strength of the unshadowed light. The light is strongest at the area of the source
// and fades smoothly to its range beyond it.
vec4 getSourceIrradiance(SourceLight light, vec2 position)
{
    float range = light.centerRange.z;
    float distance = getSourceDistance(light, position);
    if (distance >= range) {
        return vec4(0.0);
    }
    float fade = 1.0 - distance / range;
    float strength = SourceLightStrength * fade * fade;
    return vec4(light.colorShape.rgb * strength * getSourceTransmittance(light, position), strength);
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
    ivec2 node = ivec2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(node, frame.gasLightGrid.zw))) {
        return;
    }
    vec2 position = frame.gasGrid.xy + vec2(node) * frame.gasLight.z;

    // The light of a cluster comes from the objects within one cluster around it
    ivec2 lowerCluster;
    ivec2 upperCluster;
    getClusterBounds(position, frame.gasLight.y, lowerCluster, upperCluster);
    vec3 irradiance = vec3(0.0);
    for (int y = lowerCluster.y; y <= upperCluster.y; ++y) {
        for (int x = lowerCluster.x; x <= upperCluster.x; ++x) {
            irradiance += getCellLight(position, ivec2(x, y));
        }
    }
    imageStore(gasIrradianceImage, node, vec4(irradiance, 1.0));

    vec4 sourceIrradiance = vec4(0.0);
    for (int i = 0; i < frame.numSourceLights.x; ++i) {
        sourceIrradiance += getSourceIrradiance(frame.sourceLights[i], position);
    }
    imageStore(gasIrradianceImage, node + ivec2(GasIrradianceImageSize, 0), sourceIrradiance);
}
)";
}
