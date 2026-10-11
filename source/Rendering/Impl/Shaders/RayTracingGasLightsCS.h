#pragma once

#include <string_view>

namespace Shaders
{
    // Adds the cells to the gas grid, where they cast shadows, and their light to the light clusters. The light of a cell grows with its
    // energy and flashes with the activity of its neuron.
    std::string_view const RayTracingGasLightsCS = R"(
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

const int ObjectType_FreeCell = 2;
const int ObjectType_Cell = 3;

const float CellLight = 1.5;
const float FlashLight = 8.0;

void main()
{
    uint index = gl_GlobalInvocationID.x;
    if (index >= frame.counts.x) {
        return;
    }
    ObjectVertex object = objects[index];
    int objectType = (object.state >> 8) & 0xff;
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
        vec2 axisWeights = mix(1.0 - fraction, fraction, vec2(offset));
        atomicAdd(gasAccumulation[getGasNodeIndex(target) * GasAccumulationStride + 4u], uint(axisWeights.x * axisWeights.y * GasFixedPointScale + 0.5));
    }

    vec3 color = vec3(object.color[0], object.color[1], object.color[2]);
    float colorBrightness = maxComponent(color);
    vec3 hue = colorBrightness > 0.02 ? clamp(color / colorBrightness, 0.0, 1.0) : vec3(0.5);
    float brightness = min(colorBrightness, 1.0) + 0.5 * log(max(colorBrightness, 1.0));
    float flash = objectType == ObjectType_Cell ? object.highlightIntensity : 0.0;
    vec3 light = hue * (CellLight * brightness + FlashLight * flash);
    float lightLuminance = luminance(light);
    if (lightLuminance <= 0.0) {
        return;
    }

    vec2 clusterPosition = (position - frame.gasGrid.xy) / frame.gasLight.x;
    ivec2 cluster = ivec2(floor(clusterPosition));
    if (any(lessThan(cluster, ivec2(0))) || any(greaterThanEqual(cluster, frame.gasLightGrid.xy))) {
        return;
    }
    vec2 offset = clusterPosition - vec2(cluster);
    uint entry = uint(cluster.y * frame.gasLightGrid.x + cluster.x) * GasLightClusterStride;
    atomicAdd(gasLightClusters[entry], uint(lightLuminance * offset.x * GasFixedPointScale + 0.5));
    atomicAdd(gasLightClusters[entry + 1u], uint(lightLuminance * offset.y * GasFixedPointScale + 0.5));
    atomicAdd(gasLightClusters[entry + 2u], uint(lightLuminance * GasFixedPointScale + 0.5));
    atomicAdd(gasLightClusters[entry + 3u], uint(light.r * GasFixedPointScale + 0.5));
    atomicAdd(gasLightClusters[entry + 4u], uint(light.g * GasFixedPointScale + 0.5));
    atomicAdd(gasLightClusters[entry + 5u], uint(light.b * GasFixedPointScale + 0.5));
}
)";
}
