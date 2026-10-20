#pragma once

#include <string_view>

namespace Shaders
{
    // Converts the sums of the objects outside the extracted region at the nodes of the shadow grid into their extinction per world unit. The
    // gas, the tissue and the solids absorb as in the shadows of the extracted objects. The gas is blurred like the gas grid in two passes over
    // its intermediate buffer, the direction is defined in front of the source. The tissue and the solids stay sharp like the extracted
    // objects, blurred they would cast wider and darker shadows.
    std::string_view const RayTracingShadowGridCS = R"(
#ifdef VERTICAL
const ivec2 BlurDirection = ivec2(0, 1);
#else
const ivec2 BlurDirection = ivec2(1, 0);
#endif

uint getShadowGridNodeIndex(ivec2 node)
{
    return uint(node.y * frame.shadowGridSize.x + node.x);
}

float fromFixedPoint(uint sum)
{
    return float(sum) / ShadowGridFixedPointScale;
}

// rgb: extinction of the tissue and the solids, a: extinction of the gas
vec4 getNodeExtinction(ivec2 node)
{
    ShadowGridNode sums = shadowGridNodes[getShadowGridNodeIndex(node)];
    float tissueArea = fromFixedPoint(sums.tissueArea);
    vec3 tissueHueArea = vec3(fromFixedPoint(sums.tissueHueArea[0]), fromFixedPoint(sums.tissueHueArea[1]), fromFixedPoint(sums.tissueHueArea[2]));
    vec3 absorbingTissueArea = tissueArea > 0.0 ? tissueArea * getAbsorbedShare(getSaturatedHue(tissueHueArea / tissueArea)) : vec3(0.0);
    vec3 objectExtinction = TissueExtinction * absorbingTissueArea + SolidExtinction * fromFixedPoint(sums.solidArea);
    float nodeArea = frame.shadowGrid.z * frame.shadowGrid.z;
    return vec4(objectExtinction, GasExtinction * fromFixedPoint(sums.numFluids)) / nodeArea;
}

vec4 readInput(ivec2 node)
{
#ifdef VERTICAL
    return gasIntermediate[getShadowGridNodeIndex(node)];
#else
    return getNodeExtinction(node);
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
    int radius = frame.shadowGridSize.z;
    for (int offset = -radius; offset <= radius; ++offset) {
        float weight = exp(-0.5 * float(offset * offset) / (sigma * sigma));
        weightSum += weight;
        ivec2 source = node + BlurDirection * offset;
        if (any(lessThan(source, ivec2(0))) || any(greaterThanEqual(source, frame.shadowGridSize.xy))) {
            continue;
        }
        gasSum += weight * readInput(source).a;
    }
    vec4 result = vec4(readInput(node).rgb, gasSum / weightSum);
#ifdef VERTICAL
    imageStore(shadowGridImage, node, result);
#else
    gasIntermediate[getShadowGridNodeIndex(node)] = result;
#endif
}
)";
}
