#pragma once

#include <string_view>

namespace Shaders
{
    // Adds the cells to the gas grid, where they cast shadows, and their light to the light clusters. The light of a cell grows with its
    // energy. The activity of its neuron, which lets the cell itself flash, does not change it, so that the gas does not flicker.
    std::string_view const RayTracingGasLightsCS = R"(
const float CellLight = 1.5;

// The cells are provided up to at least this distance outside the view, see Engine/Impl/GeometryKernelsService.cu. Their light fades out
// towards it, so that it does not appear suddenly when cells approach the view. This does not cover lights whose range exceeds the margin
// when zoomed out, and borderless worlds hardly larger than the view, which provide the cells on each side only up to half of the remaining
// world.
const float CellCullingMargin = 10.0;

float getVisibility(vec2 position)
{
    vec2 visibleTopLeft = frame.rect.xy;
    vec2 visibleBottomRight = visibleTopLeft + frame.view.xy / frame.view.z;
    vec2 outside = max(max(visibleTopLeft - position, position - visibleBottomRight), vec2(0.0));
    return 1.0 - smoothstep(0.0, CellCullingMargin, max(outside.x, outside.y));
}

void main()
{
    uint index = gl_GlobalInvocationID.x;
    if (index >= frame.counts.x) {
        return;
    }
    ObjectVertex object = objects[index];
    int objectType = getObjectType(object);
    if (objectType != ObjectType_Cell && objectType != ObjectType_FreeCell) {
        return;
    }
    vec2 position = vec2(object.pos[0], object.pos[1]);

    vec2 node = (position - frame.gasGrid.xy) / frame.gasGrid.z;
    ivec2 base = ivec2(floor(node));
    vec2 fraction = node - vec2(base);
    for (int i = 0; i < 4; ++i) {
        ivec2 offset = ivec2(i & 1, i >> 1);
        ivec2 target = base + offset;
        if (any(lessThan(target, ivec2(0))) || any(greaterThanEqual(target, frame.gasSize.xy))) {
            continue;
        }
        atomicAdd(gasAccumulation[getGasNodeIndex(target) * GasAccumulationStride + 4u], uint(getBilinearWeight(fraction, offset) * GasFixedPointScale + 0.5));
    }

    vec3 color = vec3(object.color[0], object.color[1], object.color[2]);
    vec3 light = getHue(color) * CellLight * getBrightness(color) * getVisibility(position);
    float lightLuminance = luminance(light);
    if (lightLuminance <= 0.0) {
        return;
    }

    vec2 clusterPosition = (position - frame.gasLighting.xy) / frame.gasLight.x;
    vec2 clusterCoord = clusterPosition - 0.5;
    ivec2 lowerCluster = ivec2(floor(clusterCoord));
    vec2 clusterFraction = clusterCoord - vec2(lowerCluster);
    for (int i = 0; i < 4; ++i) {
        ivec2 offset = ivec2(i & 1, i >> 1);
        ivec2 cluster = lowerCluster + offset;
        if (any(lessThan(cluster, ivec2(0))) || any(greaterThanEqual(cluster, frame.gasLightGrid.xy))) {
            continue;
        }
        float weight = getBilinearWeight(clusterFraction, offset) * GasFixedPointScale;
        vec2 lightOffset = clusterPosition - vec2(cluster) + GasLightOffsetShift;
        uint entry = uint(cluster.y * frame.gasLightGrid.x + cluster.x) * GasLightClusterStride;
        atomicAdd(gasLightClusters[entry], uint(weight * lightLuminance * lightOffset.x + 0.5));
        atomicAdd(gasLightClusters[entry + 1u], uint(weight * lightLuminance * lightOffset.y + 0.5));
        atomicAdd(gasLightClusters[entry + 2u], uint(weight * lightLuminance + 0.5));
        atomicAdd(gasLightClusters[entry + 3u], uint(weight * light.r + 0.5));
        atomicAdd(gasLightClusters[entry + 4u], uint(weight * light.g + 0.5));
        atomicAdd(gasLightClusters[entry + 5u], uint(weight * light.b + 0.5));
    }
}
)";
}
