#pragma once

#include <string_view>

namespace Shaders
{
    // Gaussian blur of the gas grid in two passes, the sparks get a smaller blur than the gas. The horizontal pass reads the fixed point
    // sums, the vertical pass writes the densities per area into the gas image. The direction is defined in front of the source.
    std::string_view const RayTracingGasBlurCS = R"(
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

void readHorizontalInput(ivec2 node, out vec4 gas, out vec4 spark)
{
    uint entry = getGasNodeIndex(node) * GasAccumulationStride;
    gas = vec4(float(gasAccumulation[entry + 1u]), float(gasAccumulation[entry + 2u]), float(gasAccumulation[entry + 3u]), float(gasAccumulation[entry]))
        / GasFixedPointScale;
    spark = vec4(float(gasAccumulation[entry + 4u]), float(gasAccumulation[entry + 5u]), float(gasAccumulation[entry + 6u]), 0.0) / GasFixedPointScale;
}

float getGaussian(int offset, float sigma)
{
    return exp(-0.5 * float(offset * offset) / (sigma * sigma));
}

void main()
{
    ivec2 node = ivec2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(node, frame.gasSize.xy))) {
        return;
    }
    int radius = frame.gasSize.z;
    float gasSigma = max(frame.gasSpread.x, 1e-3);
    float sparkSigma = max(frame.gasSpread.z, 1e-3);
    vec4 gasSum = vec4(0.0);
    vec4 sparkSum = vec4(0.0);
    float gasWeightSum = 0.0;
    float sparkWeightSum = 0.0;
    for (int offset = -radius; offset <= radius; ++offset) {
        float gasWeight = getGaussian(offset, gasSigma);
        float sparkWeight = getGaussian(offset, sparkSigma);
        gasWeightSum += gasWeight;
        sparkWeightSum += sparkWeight;
        ivec2 source = node + BlurDirection * offset;
        if (any(lessThan(source, ivec2(0))) || any(greaterThanEqual(source, frame.gasSize.xy))) {
            continue;
        }
#ifdef VERTICAL
        uint entry = getGasNodeIndex(source) * 2u;
        gasSum += gasWeight * gasIntermediate[entry];
        sparkSum += sparkWeight * gasIntermediate[entry + 1u];
#else
        vec4 gas;
        vec4 spark;
        readHorizontalInput(source, gas, spark);
        gasSum += gasWeight * gas;
        sparkSum += sparkWeight * spark;
#endif
    }
    gasSum /= gasWeightSum;
    sparkSum /= sparkWeightSum;
#ifdef VERTICAL
    float area = frame.gasGrid.z * frame.gasGrid.z;
    imageStore(gasImage, node, gasSum / area);
    imageStore(gasImage, node + ivec2(GasImageSize, 0), sparkSum / area);
#else
    uint entry = getGasNodeIndex(node) * 2u;
    gasIntermediate[entry] = gasSum;
    gasIntermediate[entry + 1u] = sparkSum;
#endif
}
)";
}
