#pragma once

#include <string_view>

namespace Shaders
{
    // Gaussian blur of the gas grid in two passes. The horizontal pass reads the fixed point sums, the vertical pass writes the sums per area
    // into the gas image. The direction is defined in front of the source.
    std::string_view const RayTracingGasBlurCS = R"(
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

float readAccumulation(uint entry)
{
    return float(gasAccumulation[entry]) / GasFixedPointScale;
}

void readHorizontalInput(ivec2 node, out vec4 layers[GasLayers])
{
    uint entry = getGasNodeIndex(node) * GasAccumulationStride;
    layers[GasLayer_Clouds] = vec4(readAccumulation(entry + 1u), readAccumulation(entry + 2u), readAccumulation(entry + 3u), readAccumulation(entry));
    layers[GasLayer_Cells] = vec4(readAccumulation(entry + 4u), 0.0, 0.0, 0.0);
}

void main()
{
    ivec2 node = ivec2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(node, frame.gasSize.xy))) {
        return;
    }
    float sigma = max(frame.gasGrid.w, 1e-3);
    vec4 sums[GasLayers] = vec4[GasLayers](vec4(0.0), vec4(0.0));
    float weightSum = 0.0;
    int radius = frame.gasSize.z;
    for (int offset = -radius; offset <= radius; ++offset) {
        float weight = exp(-0.5 * float(offset * offset) / (sigma * sigma));
        weightSum += weight;
        ivec2 source = node + BlurDirection * offset;
        if (any(lessThan(source, ivec2(0))) || any(greaterThanEqual(source, frame.gasSize.xy))) {
            continue;
        }
        vec4 layers[GasLayers];
#ifdef VERTICAL
        uint entry = getGasNodeIndex(source) * uint(GasLayers);
        for (int layer = 0; layer < GasLayers; ++layer) {
            layers[layer] = gasIntermediate[entry + uint(layer)];
        }
#else
        readHorizontalInput(source, layers);
#endif
        for (int layer = 0; layer < GasLayers; ++layer) {
            sums[layer] += weight * layers[layer];
        }
    }
#ifdef VERTICAL
    float area = frame.gasGrid.z * frame.gasGrid.z;
    for (int layer = 0; layer < GasLayers; ++layer) {
        imageStore(gasImage, node + ivec2(layer * GasImageSize, 0), sums[layer] / (weightSum * area));
    }
#else
    uint entry = getGasNodeIndex(node) * uint(GasLayers);
    for (int layer = 0; layer < GasLayers; ++layer) {
        gasIntermediate[entry + uint(layer)] = sums[layer] / weightSum;
    }
#endif
}
)";
}
