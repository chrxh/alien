#pragma once

#include <string_view>

namespace Shaders
{
    // Computes the light of the cells at the nodes of the irradiance grid. Each light cluster shines from the center of its light, which
    // fades with the distance and is weakened by the gas and the cells on the way, so that dense gas and creatures cast shadows. Zoomed out,
    // the light spreads over a larger falloff distance near its source, while it stays the same farther away.
    std::string_view const RayTracingGasIrradianceCS = R"(
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

// Falloff distance when zoomed in, which must match LightFalloffDistance in Rendering/Impl/RayTracingRenderStep.cpp
const float LightFalloffDistance = 5.0;
const float SourceClearance = 1.5;
const int ShadowSteps = 8;
const float GasExtinction = 0.2;
const float CellExtinction = 1.0;

float getExtinction(vec2 position)
{
    return GasExtinction * sampleGasLayer(position, GasLayer_Clouds).a + CellExtinction * sampleGasLayer(position, GasLayer_Cells).x;
}

float getTransmittance(vec2 position, vec2 direction, float distance)
{
    float pathLength = distance - SourceClearance;
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

void main()
{
    ivec2 node = ivec2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(node, frame.gasLightGrid.zw))) {
        return;
    }
    vec2 position = frame.gasGrid.xy + vec2(node) * frame.gasLight.z;
    float clusterSize = frame.gasLight.x;
    float range = frame.gasLight.y;
    float falloffDistance = frame.gasLight.w;

    // The light of a cluster comes from the cells within one cluster around it. The loops only run over the clusters inside the grid, since
    // skipping the clusters outside within the loops returned no light at all for larger ranges on an NVIDIA GPU.
    ivec2 center = ivec2(floor((position - frame.gasGrid.xy) / clusterSize));
    int clusterRange = int(ceil(range / clusterSize)) + 1;
    ivec2 lowerCluster = max(center - clusterRange, ivec2(0));
    ivec2 upperCluster = min(center + clusterRange, frame.gasLightGrid.xy - 1);

    vec3 irradiance = vec3(0.0);
    for (int y = lowerCluster.y; y <= upperCluster.y; ++y) {
        for (int x = lowerCluster.x; x <= upperCluster.x; ++x) {
            ivec2 cluster = ivec2(x, y);
            uint entry = uint(cluster.y * frame.gasLightGrid.x + cluster.x) * GasLightClusterStride;
            float lightLuminance = float(gasLightClusters[entry + 2u]);
            if (lightLuminance <= 0.0) {
                continue;
            }
            vec2 lightOffset = vec2(float(gasLightClusters[entry]), float(gasLightClusters[entry + 1u])) / lightLuminance - GasLightOffsetShift;
            vec2 lightPosition = frame.gasGrid.xy + (vec2(cluster) + lightOffset) * clusterSize;
            vec2 toLight = lightPosition - position;
            float distance = length(toLight);
            if (distance >= range) {
                continue;
            }
            vec3 light = vec3(float(gasLightClusters[entry + 3u]), float(gasLightClusters[entry + 4u]), float(gasLightClusters[entry + 5u])) / GasFixedPointScale;
            float falloff = LightFalloffDistance * LightFalloffDistance / (falloffDistance * falloffDistance + distance * distance)
                * (1.0 - smoothstep(range * 0.6, range, distance));
            irradiance += light * falloff * getTransmittance(position, toLight / max(distance, 1e-4), distance);
        }
    }
    imageStore(gasIrradianceImage, node, vec4(irradiance, 1.0));
}
)";
}
