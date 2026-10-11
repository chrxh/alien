#pragma once

#include <string_view>

namespace Shaders
{
    // Computes the light of the cells at the nodes of the irradiance grid. Each light cluster shines from the center of its light, which
    // fades with the distance and is weakened by the gas and the cells on the way, so that dense gas and creatures cast shadows.
    std::string_view const RayTracingGasIrradianceCS = R"(
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

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
    ivec2 center = ivec2(floor((position - frame.gasGrid.xy) / clusterSize));
    int clusterRange = int(ceil(range / clusterSize));

    vec3 irradiance = vec3(0.0);
    for (int y = -clusterRange; y <= clusterRange; ++y) {
        for (int x = -clusterRange; x <= clusterRange; ++x) {
            ivec2 cluster = center + ivec2(x, y);
            if (any(lessThan(cluster, ivec2(0))) || any(greaterThanEqual(cluster, frame.gasLightGrid.xy))) {
                continue;
            }
            uint entry = uint(cluster.y * frame.gasLightGrid.x + cluster.x) * GasLightClusterStride;
            float lightLuminance = float(gasLightClusters[entry + 2u]);
            if (lightLuminance <= 0.0) {
                continue;
            }
            vec2 lightOffset = vec2(float(gasLightClusters[entry]), float(gasLightClusters[entry + 1u])) / lightLuminance;
            vec2 lightPosition = frame.gasGrid.xy + (vec2(cluster) + lightOffset) * clusterSize;
            vec2 toLight = lightPosition - position;
            float distance = length(toLight);
            if (distance >= range) {
                continue;
            }
            vec3 light = vec3(float(gasLightClusters[entry + 3u]), float(gasLightClusters[entry + 4u]), float(gasLightClusters[entry + 5u])) / GasFixedPointScale;
            float falloff = LightFalloffDistance * LightFalloffDistance / (LightFalloffDistance * LightFalloffDistance + distance * distance)
                * smoothstep(range, range * 0.6, distance);
            irradiance += light * falloff * getTransmittance(position, toLight / max(distance, 1e-4), distance);
        }
    }
    imageStore(gasIrradianceImage, node, vec4(irradiance, 1.0));
}
)";
}
