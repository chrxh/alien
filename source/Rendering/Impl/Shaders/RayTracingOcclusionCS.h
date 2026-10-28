#pragma once

#include <string_view>

namespace Shaders
{
    // Computes a level of an occlusion pyramid, which is defined in front of the source together with the grid of the pyramid. The pyramid of
    // the gas grid holds the extracted objects. Its first level is the occlusion per area that their spheres add to the nodes of the gas grid, see
    // Shaders/RayTracingPrepareCS.h. The pyramid of the shadow grid holds the objects outside the extracted region, its first level is the
    // shadow grid image, see Shaders/RayTracingShadowGridCS.h. Each following level is blurred and downsampled from the level before, so that
    // a beam of the light of the radiation sources finds the occlusion across its width at the level whose node distance matches it, see
    // Shaders/RayTracingGasIrradianceCS.h.
    std::string_view const RayTracingOcclusionCS = R"(
#ifdef SHADOW_GRID
const int GridImageSize = ShadowGridImageSize;

ivec2 getGridSize()
{
    return frame.shadowGridSize.xy;
}

vec4 loadTexel(ivec2 texel)
{
    return imageLoad(shadowGridImage, texel);
}

void storeTexel(ivec2 texel, vec4 value)
{
    imageStore(shadowGridImage, texel, value);
}
#else
const int GridImageSize = GasImageSize;

ivec2 getGridSize()
{
    return frame.gasSize.xy;
}

vec4 loadTexel(ivec2 texel)
{
    return imageLoad(occlusionImage, texel);
}

void storeTexel(ivec2 texel, vec4 value)
{
    imageStore(occlusionImage, texel, value);
}
#endif

#if OCCLUSION_LEVEL == 0
// A node takes the three nodes of the gas grid around it along each axis with the weights 1, 2, 1, so that beams sampling the first level
// once per node, or beams lying one node apart, take up every object about evenly. The nodes outside the gas grid count as empty.
const float BlurWeights[3] = float[3](0.25, 0.5, 0.25);

vec4 getLevelValue(ivec2 node)
{
    vec3 sum = vec3(0.0);
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            ivec2 gasNode = node + ivec2(x, y);
            if (all(greaterThanEqual(gasNode, ivec2(0))) && all(lessThan(gasNode, frame.gasSize.xy))) {
                uint entry = getGasNodeIndex(gasNode) * OcclusionStride;
                vec3 value = vec3(float(occlusionAccumulation[entry]), float(occlusionAccumulation[entry + 1u]), float(occlusionAccumulation[entry + 2u]));
                sum += BlurWeights[x + 1] * BlurWeights[y + 1] * value;
            }
        }
    }
    return vec4(sum / (OcclusionFixedPointScale * frame.gasGrid.z * frame.gasGrid.z), 0.0);
}
#else
// A node takes the four nodes of the level before around it along each axis with the weights 1, 3, 3, 1, so that the occlusion of an object
// moves smoothly with it at every level. The nodes outside the level before count as empty.
const float DownsamplingWeights[4] = float[4](0.125, 0.375, 0.375, 0.125);

vec4 getLevelValue(ivec2 node)
{
    int previousLevel = OCCLUSION_LEVEL - 1;
    ivec2 previousStart = getPyramidLevelStart(GridImageSize, previousLevel);
    ivec2 previousSize = getPyramidLevelSize(getGridSize(), previousLevel);
    vec4 result = vec4(0.0);
    for (int y = 0; y < 4; ++y) {
        for (int x = 0; x < 4; ++x) {
            ivec2 previousNode = 2 * node + ivec2(x, y) - 1;
            if (all(greaterThanEqual(previousNode, ivec2(0))) && all(lessThan(previousNode, previousSize))) {
                result += DownsamplingWeights[x] * DownsamplingWeights[y] * loadTexel(previousStart + previousNode);
            }
        }
    }
    return result;
}
#endif

void main()
{
    ivec2 node = ivec2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(node, getPyramidLevelSize(getGridSize(), OCCLUSION_LEVEL)))) {
        return;
    }
    storeTexel(getPyramidLevelStart(GridImageSize, OCCLUSION_LEVEL) + node, getLevelValue(node));
}
)";
}
