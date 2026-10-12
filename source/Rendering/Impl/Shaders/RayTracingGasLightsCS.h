#pragma once

#include <string_view>

namespace Shaders
{
    // Adds the cells to the gas grid, where they cast shadows, and their light to the light clusters. The light of a cell grows with its
    // energy. The activity of its neuron, which lets the cell itself flash, does not change it, so that the gas does not flicker. Every
    // object in a region with a high energy density adds its share of the radiation of the region to the energy light.
    std::string_view const RayTracingGasLightsCS = R"(
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

const float CellLight = 1.5;

// The cells are provided up to at least this distance outside the view, see Engine/Impl/GeometryKernelsService.cu. Their light fades out
// towards it, so that it does not appear suddenly when cells approach the view. This does not cover lights whose range exceeds the margin
// when zoomed out, and cells to the left of and above the view in borderless worlds, which are not provided at all.
const float CellCullingMargin = 10.0;

float getVisibility(vec2 position)
{
    vec2 visibleTopLeft = frame.rect.xy;
    vec2 visibleBottomRight = visibleTopLeft + frame.view.xy / frame.view.z;
    vec2 outside = max(max(visibleTopLeft - position, position - visibleBottomRight), vec2(0.0));
    return 1.0 - smoothstep(0.0, CellCullingMargin, max(outside.x, outside.y));
}

vec3 getHue(vec3 color)
{
    float colorBrightness = maxComponent(color);
    return colorBrightness > 0.02 ? clamp(color / colorBrightness, 0.0, 1.0) : vec3(0.5);
}

// The light is added bilinearly to the four clusters around its position
void addLight(vec2 position, vec3 light, uint firstEntry, float fixedPointScale)
{
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
        vec2 axisWeights = mix(1.0 - clusterFraction, clusterFraction, vec2(offset));
        float weight = axisWeights.x * axisWeights.y * fixedPointScale;
        vec2 lightOffset = clusterPosition - vec2(cluster) + GasLightOffsetShift;
        uint entry = uint(cluster.y * frame.gasLightGrid.x + cluster.x) * GasLightClusterStride + firstEntry;
        atomicAdd(gasLightClusters[entry], uint(weight * lightLuminance * lightOffset.x + 0.5));
        atomicAdd(gasLightClusters[entry + 1u], uint(weight * lightLuminance * lightOffset.y + 0.5));
        atomicAdd(gasLightClusters[entry + 2u], uint(weight * lightLuminance + 0.5));
        atomicAdd(gasLightClusters[entry + 3u], uint(weight * light.r + 0.5));
        atomicAdd(gasLightClusters[entry + 4u], uint(weight * light.g + 0.5));
        atomicAdd(gasLightClusters[entry + 5u], uint(weight * light.b + 0.5));
    }
}

void main()
{
    uint index = gl_GlobalInvocationID.x;
    if (index >= frame.counts.x) {
        return;
    }
    ObjectVertex object = objects[index];
    vec2 position = vec2(object.pos[0], object.pos[1]);
    vec3 color = vec3(object.color[0], object.color[1], object.color[2]);
    // The radiation per area grows with the logarithm of the energy density above the threshold, so that it sets in softly and the
    // regions with very much energy do not outshine everything. Each object contributes its share of the energy.
    if (isEnergyLightActive()) {
        float density = getEnergyDensity(position);
        if (density > frame.energyLight.y) {
            float emission = frame.energyLight.x * log(density / frame.energyLight.y) * getObjectEnergy(object) / density * getVisibility(position);
            addLight(position, getHue(color) * emission, GasLightEntry_Energy, EnergyLightFixedPointScale);
        }
    }

    int objectType = (object.state >> 8) & 0xff;
    if (objectType != ObjectType_Cell && objectType != ObjectType_FreeCell) {
        return;
    }

    vec2 node = (position - frame.gasGrid.xy) / frame.gasGrid.z;
    ivec2 base = ivec2(floor(node));
    vec2 fraction = node - vec2(base);
    for (int i = 0; i < 4; ++i) {
        ivec2 offset = ivec2(i & 1, i >> 1);
        ivec2 target = base + offset;
        if (any(lessThan(target, ivec2(0))) || any(greaterThanEqual(target, frame.gasSize.xy))) {
            continue;
        }
        vec2 axisWeights = mix(1.0 - fraction, fraction, vec2(offset));
        atomicAdd(gasAccumulation[getGasNodeIndex(target) * GasAccumulationStride + 4u], uint(axisWeights.x * axisWeights.y * GasFixedPointScale + 0.5));
    }

    float colorBrightness = maxComponent(color);
    float brightness = min(colorBrightness, 1.0) + 0.5 * log(max(colorBrightness, 1.0));
    addLight(position, getHue(color) * CellLight * brightness * getVisibility(position), GasLightEntry_Cells, GasFixedPointScale);
}
)";
}
