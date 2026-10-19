#pragma once

#include <string_view>

namespace Shaders
{
    // Declarations shared by the compute shaders of the ray tracing, which shows the creatures as living tissue under a microscope. The
    // polygons of connected cells form translucent bodies, the cells bulge slightly out of them, free connections become thin strands.
    // Everything is computed without random samples, so that the frames are free of noise.
    // Pixel coordinates of the internal images start at the upper left corner of the view, i.e. their rows run along the world y axis. The
    // version, the work group size and the sizes shared with Rendering/Impl/RayTracingResources.cpp are declared before, see createPipeline.
    std::string_view const RayTracingCommonCS = R"(
const float CameraHeight = 20.0;

// Visible sizes of the tissue close up in world units
const float CellRadius = 0.34;
const float FreeCellRadius = 0.29;
const float SolidRadius = 0.4;
const float StrandHalfWidth = 0.165;
const float FillOffset = 0.105;

// Zoomed out, the tissue approaches the thin lines and dots of the rasterized view, but stays at least as large as a pixel
const float FarStrandHalfWidth = 0.075;
const float FarDotRadius = 0.25;
const float MinStrandHalfWidthInPixels = 1.0;
const float MinDotRadiusInPixels = 1.75;
const float MinFillOffsetInPixels = 0.25;

// The contributions of connected objects add up to a density whose threshold forms the outline. A contribution vanishes at the range
// of its kernel, which exceeds its visible size, so that connected objects blend smoothly.
const float DensityThreshold = 0.33;
const float CellWeight = 0.8;
const float StrandWeight = 0.75;
const float FillWeight = 1.0;

// Relative deviation of the cells from a round shape
const float CellShapeVariation = 0.1;

const uint Kind_Sphere = 1u;
const uint Kind_Capsule = 2u;
const uint Kind_Triangle = 3u;
const uint NoEntry = 0xffffffffu;

const float Material_Tissue = 0.0;
const float Material_Solid = 1.0;
const float Material_Inactive = 2.0;

const int ObjectType_Solid = 0;
const int ObjectType_Fluid = 1;
const int ObjectType_FreeCell = 2;
const int ObjectType_Cell = 3;

struct ObjectVertex
{
    float pos[3];
    float color[3];
    int state;
    float highlightIntensity;
    float activity;
};

struct Sphere
{
    vec4 centerRadius;  // xy: center, w: radius
    vec4 hueMaterial;   // rgb: hue, w: material
    vec4 details;       // x: brightness of the color, y: random seed that stays with the object, z: 1 if isolated or at the end of a chain
    vec4 neuron;        // x: 1 for a cell with a neuron, y: activity of the neuron
};

struct Aabb
{
    float minX;
    float minY;
    float minZ;
    float maxX;
    float maxY;
    float maxZ;
};

struct FluidParticle
{
    float pos[3];
    float color[3];
    float glow;
    float vel[2];
};

// A radiation source that shines from its circular or rectangular area
struct SourceLight
{
    vec4 centerRangeShape;  // xy: world position of the center, z: range of the light beyond the area, w: 0 for a circular, 1 for a
                            // rectangular area
    vec4 color;             // rgb: color times brightness
    vec4 extent;            // xy: radius of a circular area in both components or half the size of a rectangular area
};

layout(set = 0, binding = 0, std140) uniform FrameConstants
{
    vec4 view;        // xy: size in pixels, z: pixels per world unit, w: screen pixels per world unit
    vec4 rect;        // xy: world position of the upper left corner, zw: jitter of the rays in pixels
    uvec4 counts;     // x: number of objects, y: number of connections, z: number of triangles
    vec4 tissue;      // x: 1 close up, 0 zoomed out, y: visibility of the cell details, z: world size of a pixel, w: visibility of the tissue
    vec4 parameters;  // x: scale of the background color, y: 1 for anti-aliasing within the frame, z: weight of the frame in the average with
                      // the previous frames, 1 if there is no average, w: visibility of the structure of the solids
    vec4 gasGrid;     // xy: world position of the first node of the gas grid, z: distance of the nodes, w: standard deviation of the blur in nodes
    ivec4 gasSize;    // xy: number of nodes of the gas grid, z: blur radius in nodes, w: number of fluid particles
    vec4 gasLight;    // x: size of the light clusters in world units, y: range of the lights, z: distance of the nodes of the irradiance grid,
                      // w: falloff distance of the lights
    ivec4 gasLightGrid;  // xy: number of light clusters, zw: number of nodes of the irradiance grid
    vec4 gasLighting;  // xy: world position of the first light cluster, z: strength of the light of the cells in the gas
    vec4 world;        // xy: size of the world, z: 1 for borderless rendering
    ivec4 numSourceLights;  // x: number of source lights
    SourceLight sourceLights[MaxSourceLights];
} frame;

layout(set = 0, binding = 1) uniform accelerationStructureEXT scene;
layout(set = 0, binding = 2, std430) readonly buffer ObjectBuffer { ObjectVertex objects[]; };
layout(set = 0, binding = 3, std430) readonly buffer LineIndexBuffer { uint lineIndices[]; };
layout(set = 0, binding = 4, std430) readonly buffer TriangleIndexBuffer { uint triangleIndices[]; };
layout(set = 0, binding = 5, std430) buffer SphereBuffer { Sphere spheres[]; };
layout(set = 0, binding = 6, std430) buffer SphereAabbBuffer { Aabb sphereAabbs[]; };
layout(set = 0, binding = 7, std430) buffer CapsuleAabbBuffer { Aabb capsuleAabbs[]; };
layout(set = 0, binding = 8, std430) buffer TriangleAabbBuffer { Aabb triangleAabbs[]; };
layout(set = 0, binding = 9, rgba16f) uniform image2D sceneColor;
layout(set = 0, binding = 10, rgba16f) uniform image2D antiAliased;
layout(set = 0, binding = 11) uniform sampler2D previousAntiAliased;
layout(set = 0, binding = 12) uniform sampler2D groundTexture;
layout(set = 0, binding = 13) uniform sampler2D gasSampler;
layout(set = 0, binding = 14) uniform sampler2D sceneColorSampler;
layout(set = 0, binding = 15, std430) readonly buffer FluidParticleBuffer { FluidParticle fluidParticles[]; };
layout(set = 0, binding = 16, std430) buffer GasAccumulationBuffer { uint gasAccumulation[]; };
layout(set = 0, binding = 17, std430) buffer GasIntermediateBuffer { vec4 gasIntermediate[]; };
layout(set = 0, binding = 18, rgba16f) uniform image2D gasImage;
layout(set = 0, binding = 19) uniform sampler2D energyParticleTexture;
layout(set = 0, binding = 20, std430) buffer GasLightClusterBuffer { uint gasLightClusters[]; };
layout(set = 0, binding = 21, rgba16f) uniform image2D gasIrradianceImage;
layout(set = 0, binding = 22) uniform sampler2D gasIrradianceSampler;

//*******************
//* General helpers
//*******************

bool isInView(ivec2 pixel)
{
    return pixel.x >= 0 && pixel.y >= 0 && pixel.x < int(frame.view.x) && pixel.y < int(frame.view.y);
}

// Pixel positions are continuous, the center of the upper left pixel lies at (0.5, 0.5)
vec2 pixelToWorld(vec2 pixelPosition)
{
    return frame.rect.xy + pixelPosition / frame.view.z;
}

vec2 worldToPixel(vec2 world)
{
    return (world - frame.rect.xy) * frame.view.z;
}

// Texture coordinates of the textures of the render graph, whose rows run from the bottom to the top
vec2 worldToGraphTextureCoord(vec2 world)
{
    vec2 coord = worldToPixel(world) / frame.view.xy;
    return vec2(coord.x, 1.0 - coord.y);
}

float luminance(vec3 color)
{
    return dot(color, vec3(0.2126, 0.7152, 0.0722));
}

float maxComponent(vec3 value)
{
    return max(value.x, max(value.y, value.z));
}

float minComponent(vec3 value)
{
    return min(value.x, min(value.y, value.z));
}

float hash(vec2 position)
{
    vec3 mixed = fract(vec3(position.xyx) * 0.1031);
    mixed += dot(mixed, mixed.yzx + 33.33);
    return fract((mixed.x + mixed.y) * mixed.z);
}

float valueNoise(vec2 position)
{
    vec2 cell = floor(position);
    vec2 offset = fract(position);
    vec2 weight = offset * offset * (3.0 - 2.0 * offset);
    return mix(
        mix(hash(cell), hash(cell + vec2(1.0, 0.0)), weight.x), mix(hash(cell + vec2(0.0, 1.0)), hash(cell + vec2(1.0, 1.0)), weight.x), weight.y);
}

// Weight of the corner at the offset when a value is distributed bilinearly to the four corners around a position with the fraction
float getBilinearWeight(vec2 fraction, ivec2 offset)
{
    vec2 axisWeights = mix(1.0 - fraction, fraction, vec2(offset));
    return axisWeights.x * axisWeights.y;
}

int getObjectType(ObjectVertex object)
{
    return (object.state >> 8) & 0xff;
}

// The color of an object scales with its energy, the hue does not
vec3 getHue(vec3 color)
{
    float colorBrightness = maxComponent(color);
    return colorBrightness > 0.02 ? clamp(color / colorBrightness, 0.0, 1.0) : vec3(0.5);
}

// Objects with much energy have colors far above 1, which would outshine their structure
float getBrightness(vec3 color)
{
    float colorBrightness = maxComponent(color);
    return min(colorBrightness, 1.0) + 0.5 * log(max(colorBrightness, 1.0));
}

// The tissue shows the hue of its objects slightly desaturated and absorbs light mainly in the complementary colors
const float TissueSaturation = 0.9;
const float TissueHueTransmission = 0.85;

vec3 getSaturatedHue(vec3 hue)
{
    return mix(vec3(luminance(hue)), hue, TissueSaturation);
}

// Share of each color component that the tissue absorbs
vec3 getAbsorbedShare(vec3 saturatedHue)
{
    return vec3(1.0) - TissueHueTransmission * saturatedHue;
}

//*******************
//* Tissue geometry
//*******************

float getPixelSize()
{
    return frame.tissue.z;
}

float getCloseUpWeight()
{
    return frame.tissue.x;
}

float getSolidDetailVisibility()
{
    return frame.parameters.w;
}

// Ratio of the visible size of a single contribution to the range of its kernel
float getContourFraction(float weight)
{
    return sqrt(1.0 - pow(DensityThreshold / weight, 1.0 / 3.0));
}

// Connected solids appear as smooth areas unless their structure is visible. Their strands are then as wide as their particles, which do not
// add to the density of their strands and polygons, so that the particles do not bulge out of the outlines.
float getSolidSmoothing(Sphere sphere)
{
    return sphere.hueMaterial.w == Material_Solid && sphere.details.z < 0.5 ? 1.0 - getSolidDetailVisibility() : 0.0;
}

float getStrandHalfWidth(float solidSmoothing)
{
    float closeUpHalfWidth = mix(StrandHalfWidth, SolidRadius, solidSmoothing);
    return max(mix(FarStrandHalfWidth, closeUpHalfWidth, getCloseUpWeight()), MinStrandHalfWidthInPixels * getPixelSize());
}

// Zoomed out, connected cells shrink to the width of the strands, while isolated cells and the ends of chains remain dots
float getCellRadius(Sphere sphere)
{
    float farRadius = sphere.details.z > 0.5 ? max(FarDotRadius, MinDotRadiusInPixels * getPixelSize()) : getStrandHalfWidth(0.0);
    float radius = mix(farRadius, max(sphere.centerRadius.w, farRadius), getCloseUpWeight());
    return mix(radius, getStrandHalfWidth(1.0), getSolidSmoothing(sphere));
}

float getCellKernelRadius(Sphere sphere)
{
    return getCellRadius(sphere) / getContourFraction(CellWeight);
}

// The outline of a cell deviates from a circle by up to this range
float getMaxCellKernelRadius(Sphere sphere)
{
    return getCellKernelRadius(sphere) * (1.0 + CellShapeVariation);
}

float getStrandSolidSmoothing(Sphere a, Sphere b)
{
    return min(getSolidSmoothing(a), getSolidSmoothing(b));
}

float getStrandKernelRadius(float solidSmoothing)
{
    return getStrandHalfWidth(solidSmoothing) / getContourFraction(StrandWeight);
}

float getFillFalloff()
{
    float offset = max(FillOffset * getCloseUpWeight(), MinFillOffsetInPixels * getPixelSize());
    return offset / getContourFraction(FillWeight);
}

void getCapsuleSpheres(uint capsule, out uint sphereA, out uint sphereB)
{
    sphereA = lineIndices[2u * capsule];
    sphereB = lineIndices[2u * capsule + 1u];
}

void getTriangleSpheres(uint triangle, out uint sphereA, out uint sphereB, out uint sphereC)
{
    sphereA = triangleIndices[3u * triangle];
    sphereB = triangleIndices[3u * triangle + 1u];
    sphereC = triangleIndices[3u * triangle + 2u];
}

//*************************
//* Background
//*************************

// The tone mapping of Shaders/ToneMappingFS.h for non-negative colors
vec3 toneMapping(vec3 color)
{
    vec3 mapped = clamp((color * (2.51 * color + 0.03)) / (color * (2.43 * color + 0.59) + 0.14), 0.0, 1.0);
    return pow(mapped, vec3(1.0 / 1.6));
}

// The background color is meant as the color after the tone mapping, so that empty regions keep their look
vec3 inverseToneMapping(vec3 color)
{
    vec3 mapped = min(pow(clamp(color, 0.0, 1.0), vec3(1.6)), vec3(0.98));
    vec3 qa = mapped * 2.43 - 2.51;
    vec3 qb = mapped * 0.59 - 0.03;
    vec3 qc = mapped * 0.14;
    return (-qb - sqrt(max(qb * qb - 4.0 * qa * qc, vec3(0.0)))) / (2.0 * qa);
}

vec3 getBackgroundRadiance(vec2 world)
{
    return inverseToneMapping(texture(groundTexture, worldToGraphTextureCoord(world)).rgb) * frame.parameters.x;
}

)"
                                                R"(
//*******************************
//* Gas of the fluid particles
//*******************************

// The particles accumulate fixed point sums at the nodes of the gas grid, which are blurred into the layers of the gas image side by side.
// The GasAccumulationStride sums per node are the number of particles, their color sum and the number of cells, which cast shadows in the
// gas.
const float GasFixedPointScale = 1024.0;

// The light of the cells is collected in clusters of the light cluster grid with the sums of the luminance weighted offsets to the cluster
// corner, the luminance and the color. Each cell contributes bilinearly to the four clusters around it, so that the light changes smoothly
// when cells move from one cluster to another. The offsets are shifted by one cluster to stay positive. The irradiance image holds the light
// of the cells and the light of the radiation sources side by side.
const float GasLightOffsetShift = 1.0;

// Clouds: color sum and number of particles. Cells: number of cells in x.
const int GasLayer_Clouds = 0;
const int GasLayer_Cells = 1;

bool isFluid(FluidParticle particle)
{
    return particle.pos[2] >= 0.5;
}

uint getGasNodeIndex(ivec2 node)
{
    return uint(node.y * frame.gasSize.x + node.x);
}

// Fast linear sampling within a region of a texture, which starts at the given texel and has the given size in texels
vec4 sampleRegionLinear(sampler2D image, vec2 texel, vec2 regionStart, vec2 regionSize, vec2 imageSize)
{
    return texture(image, (regionStart + clamp(texel, vec2(0.5), regionSize - 0.5)) / imageSize);
}

// Smooth sampling with a cubic B-spline, which avoids the kinks of the linear interpolation of coarse grids. Four linear samples combine the
// sixteen weights.
vec4 sampleRegionCubic(sampler2D image, vec2 texel, vec2 regionStart, vec2 regionSize, vec2 imageSize)
{
    vec2 coord = texel - 0.5;
    vec2 index = floor(coord);
    vec2 fraction = coord - index;
    vec2 inverse = 1.0 - fraction;
    vec2 weight0 = inverse * inverse * inverse / 6.0;
    vec2 weight1 = (4.0 - 6.0 * fraction * fraction + 3.0 * fraction * fraction * fraction) / 6.0;
    vec2 weight3 = fraction * fraction * fraction / 6.0;
    vec2 weight2 = 1.0 - weight0 - weight1 - weight3;
    vec2 lowerWeight = weight0 + weight1;
    vec2 upperWeight = weight2 + weight3;
    vec2 lowerTexel = index - 0.5 + weight1 / lowerWeight;
    vec2 upperTexel = index + 1.5 + weight3 / upperWeight;
    return lowerWeight.y * (lowerWeight.x * sampleRegionLinear(image, vec2(lowerTexel.x, lowerTexel.y), regionStart, regionSize, imageSize)
                            + upperWeight.x * sampleRegionLinear(image, vec2(upperTexel.x, lowerTexel.y), regionStart, regionSize, imageSize))
        + upperWeight.y * (lowerWeight.x * sampleRegionLinear(image, vec2(lowerTexel.x, upperTexel.y), regionStart, regionSize, imageSize)
                           + upperWeight.x * sampleRegionLinear(image, vec2(upperTexel.x, upperTexel.y), regionStart, regionSize, imageSize));
}

vec2 getGasTexel(vec2 world)
{
    return (world - frame.gasGrid.xy) / frame.gasGrid.z + 0.5;
}

vec2 getGasLayerStart(int layer)
{
    return vec2(float(layer * GasImageSize), 0.0);
}

vec2 getGasImageSize()
{
    return vec2(GasLayers * GasImageSize, GasImageSize);
}

// Fast sampling for the many samples along the paths to the lights
vec4 sampleGasLayer(vec2 world, int layer)
{
    return sampleRegionLinear(gasSampler, getGasTexel(world), getGasLayerStart(layer), vec2(frame.gasSize.xy), getGasImageSize());
}

// Smooth sampling for the visible gas
vec4 sampleGasLayerSmooth(vec2 world, int layer)
{
    return sampleRegionCubic(gasSampler, getGasTexel(world), getGasLayerStart(layer), vec2(frame.gasSize.xy), getGasImageSize());
}

vec4 sampleIrradiance(vec2 world, float regionStart)
{
    vec2 texel = (world - frame.gasGrid.xy) / frame.gasLight.z + 0.5;
    vec2 imageSize = vec2(2 * GasIrradianceImageSize, GasIrradianceImageSize);
    return sampleRegionCubic(gasIrradianceSampler, texel, vec2(regionStart, 0.0), vec2(frame.gasLightGrid.zw), imageSize);
}

// Light of the cells that reaches a position in the gas
vec3 sampleGasIrradiance(vec2 world)
{
    return sampleIrradiance(world, 0.0).rgb;
}
)"
                                                R"(
//*******************************
//* Light of the radiation sources
//*******************************

// The shadow rays to the radiation sources stand for beams that lead from the parts of a node of the irradiance grid to the parts of the
// shining area side by side, see Shaders/RayTracingGasIrradianceCS.h. At least this many rays start from a node.
const int NodeRayStarts = 4;

// The beams may be as wide as the parts of a node, but at least half a world unit, so that large areas need not too many rays when zoomed in
const float MinBeamWidthLimit = 0.5;

float getMaxBeamWidth()
{
    return max(sqrt(2.0) * frame.gasLight.z / float(NodeRayStarts), MinBeamWidthLimit);
}

bool isSourceLightActive()
{
    return frame.numSourceLights.x > 0;
}

bool isCircular(SourceLight light)
{
    return light.centerRangeShape.w < 0.5;
}

vec2 getClosestSourcePoint(SourceLight light, vec2 position)
{
    vec2 center = light.centerRangeShape.xy;
    vec2 offset = position - center;
    if (isCircular(light)) {
        float distance = length(offset);
        return distance > light.extent.x ? center + offset * (light.extent.x / distance) : position;
    }
    return center + clamp(offset, -light.extent.xy, light.extent.xy);
}

// Center of the periodic copy of a source next to the position
vec2 getNearestSourceCenter(SourceLight light, vec2 position)
{
    vec2 world = frame.world.xy;
    return light.centerRangeShape.xy + world * round((position - light.centerRangeShape.xy) / world);
}

// The bounding boxes of the objects are enlarged by half the width of the widest beam, so that the beams meet the objects next to them
float getShadowBeamMargin()
{
    return isSourceLightActive() ? 0.5 * getMaxBeamWidth() : 0.0;
}

// The light of each source approaches its brightness close to it, see Shaders/RayTracingGasIrradianceCS.h
vec3 getSourceLight(vec2 world)
{
    return max(sampleIrradiance(world, float(GasIrradianceImageSize)).rgb, vec3(0.0));
}

// Without borderless rendering, the space outside the world stays dark
bool isInWorld(vec2 world)
{
    return frame.world.z > 0.5 || (all(greaterThanEqual(world, vec2(0.0))) && all(lessThanEqual(world, frame.world.xy)));
}

// The energy particles and the cores of glowing fluids keep the look of the rasterized view
vec3 getEnergyParticleRadiance(vec2 world)
{
    return texture(energyParticleTexture, worldToGraphTextureCoord(world)).rgb;
}
)";
}
