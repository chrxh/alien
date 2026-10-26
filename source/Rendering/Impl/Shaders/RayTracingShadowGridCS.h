#pragma once

#include <string_view>

namespace Shaders
{
    // Converts the sums of the objects outside the extracted region at the nodes of the shadow grid into the occlusion of the tissue and the
    // solids per area, see Shaders/RayTracingCommonCS.h, and the extinction of the gas per world unit. These objects are only known by their
    // areas, which count as cells or particles of solids, so that they shade the light like the extracted objects. The gas is blurred like the
    // gas grid in two passes over its intermediate buffer, the direction is defined in front of the source. The occlusion is only blurred
    // slightly, the pyramid of the shadow grid blurs it further for wide beams, see Shaders/RayTracingOcclusionCS.h.
    std::string_view const RayTracingShadowGridCS = R"(
#ifdef VERTICAL
const ivec2 BlurDirection = ivec2(0, 1);
#else
const ivec2 BlurDirection = ivec2(1, 0);
#endif

// The occlusion is blurred slightly, as the first level of the pyramid of the gas grid, see Shaders/RayTracingOcclusionCS.h
const float OcclusionBlurWeights[3] = float[3](0.25, 0.5, 0.25);

uint getShadowGridNodeIndex(ivec2 node)
{
    return uint(node.y * frame.shadowGridSize.x + node.x);
}

float fromFixedPoint(uint sum)
{
    return float(sum) / ShadowGridFixedPointScale;
}

// Occlusion per area of disks of the given radius
vec3 getDiskOcclusionPerArea(float radius, vec3 absorption)
{
    return getOcclusion(2.0 * Pi * radius, Pi * radius * radius, absorption) / (Pi * radius * radius);
}

// rgb: occlusion per area of the tissue and the solids, a: extinction of the gas
vec4 getNodeOcclusion(ivec2 node)
{
    ShadowGridNode sums = shadowGridNodes[getShadowGridNodeIndex(node)];
    float tissueArea = fromFixedPoint(sums.tissueArea);
    vec3 tissueHueArea = vec3(fromFixedPoint(sums.tissueHueArea[0]), fromFixedPoint(sums.tissueHueArea[1]), fromFixedPoint(sums.tissueHueArea[2]));
    vec3 tissueOcclusion = vec3(0.0);
    if (tissueArea > 0.0) {
        vec3 tissueAbsorption = TissueExtinction * getAbsorbedShare(getSaturatedHue(tissueHueArea / tissueArea));
        tissueOcclusion = tissueArea * getDiskOcclusionPerArea(CellRadius, tissueAbsorption);
    }
    vec3 solidOcclusion = fromFixedPoint(sums.solidArea) * getDiskOcclusionPerArea(SolidRadius, vec3(SolidExtinction));
    float nodeArea = frame.shadowGrid.z * frame.shadowGrid.z;
    return vec4(tissueOcclusion + solidOcclusion, GasExtinction * fromFixedPoint(sums.numFluids)) / nodeArea;
}

vec4 readInput(ivec2 node)
{
#ifdef VERTICAL
    return gasIntermediate[getShadowGridNodeIndex(node)];
#else
    return getNodeOcclusion(node);
#endif
}

void main()
{
    ivec2 node = ivec2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(node, frame.shadowGridSize.xy))) {
        return;
    }
    float sigma = max(frame.shadowGrid.w, 1e-3);
    float gasSum = 0.0;
    float weightSum = 0.0;
    vec3 occlusionSum = vec3(0.0);
    int radius = max(frame.shadowGridSize.z, 1);
    for (int offset = -radius; offset <= radius; ++offset) {
        float weight = exp(-0.5 * float(offset * offset) / (sigma * sigma));
        weightSum += weight;
        ivec2 source = node + BlurDirection * offset;
        if (any(lessThan(source, ivec2(0))) || any(greaterThanEqual(source, frame.shadowGridSize.xy))) {
            continue;
        }
        vec4 value = readInput(source);
        gasSum += weight * value.a;
        if (abs(offset) <= 1) {
            occlusionSum += OcclusionBlurWeights[offset + 1] * value.rgb;
        }
    }
    vec4 result = vec4(occlusionSum, gasSum / weightSum);
#ifdef VERTICAL
    imageStore(shadowGridImage, node, result);
#else
    gasIntermediate[getShadowGridNodeIndex(node)] = result;
#endif
}
)";
}
