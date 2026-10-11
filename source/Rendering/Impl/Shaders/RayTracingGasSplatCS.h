#pragma once

#include <string_view>

namespace Shaders
{
    // Distributes each fluid particle bilinearly over the four surrounding nodes of the gas grid. Glowing particles also enter the sparks,
    // which stay sharper than the gas, and brighten the gas around them. The sums are fixed point numbers, so that the atomic additions give
    // the same result in any order.
    std::string_view const RayTracingGasSplatCS = R"(
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

const float SparkLightInGas = 4.0;

void main()
{
    uint index = gl_GlobalInvocationID.x;
    if (index >= uint(frame.gasSize.w)) {
        return;
    }
    FluidParticle particle = fluidParticles[index];
    vec2 node = (vec2(particle.pos[0], particle.pos[1]) - frame.gasGrid.xy) / frame.gasGrid.z;
    ivec2 base = ivec2(floor(node));
    vec2 fraction = node - vec2(base);
    float glow = clamp(particle.glow, 0.0, 1.0);
    vec3 particleColor = vec3(particle.color[0], particle.color[1], particle.color[2]);
    vec3 color = particleColor * (1.0 + SparkLightInGas * glow);
    vec3 spark = particleColor * glow;
    for (int i = 0; i < 4; ++i) {
        ivec2 offset = ivec2(i & 1, i >> 1);
        ivec2 target = base + offset;
        if (any(lessThan(target, ivec2(0))) || any(greaterThanEqual(target, frame.gasSize.xy))) {
            continue;
        }
        vec2 axisWeights = mix(1.0 - fraction, fraction, vec2(offset));
        float weight = axisWeights.x * axisWeights.y * GasFixedPointScale;
        uint entry = getGasNodeIndex(target) * GasAccumulationStride;
        atomicAdd(gasAccumulation[entry], uint(weight + 0.5));
        atomicAdd(gasAccumulation[entry + 1u], uint(weight * color.r + 0.5));
        atomicAdd(gasAccumulation[entry + 2u], uint(weight * color.g + 0.5));
        atomicAdd(gasAccumulation[entry + 3u], uint(weight * color.b + 0.5));
        if (particle.glow > 0.0) {
            atomicAdd(gasAccumulation[entry + 4u], uint(weight * spark.r + 0.5));
            atomicAdd(gasAccumulation[entry + 5u], uint(weight * spark.g + 0.5));
            atomicAdd(gasAccumulation[entry + 6u], uint(weight * spark.b + 0.5));
        }
    }
}
)";
}
