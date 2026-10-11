#pragma once

#include <string_view>

namespace Shaders
{
    // Declarations shared by the compute shaders of the ray tracing, which shows the creatures as living tissue under a microscope. The
    // polygons of connected cells form translucent bodies, the cells bulge slightly out of them, free connections become thin strands.
    // Everything is computed without random samples, so that the frames are free of noise.
    // Pixel coordinates of the internal images start at the upper left corner of the view, i.e. their rows run along the world y axis.
    std::string_view const RayTracingCommonCS = R"(
#version 460
#extension GL_EXT_ray_query : require

#define PI 3.14159265

const float CameraHeight = 20.0;
const float EmissionToRadiance = 0.6;
const float MinLightRadiance = 0.06;

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

struct ObjectVertex
{
    float pos[3];
    float color[3];
    int state;
    float highlightIntensity;
};

struct Sphere
{
    vec4 centerRadius;  // xyz: center, w: radius
    vec4 hueMaterial;   // rgb: hue, w: material
    vec4 emission;      // rgb: glow, w: brightness of the color
    vec4 details;       // x: cell type, y: random seed that stays with the object, z: 1 if isolated or at the end of a chain
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
};

layout(set = 0, binding = 0, std140) uniform FrameConstants
{
    vec4 view;        // xy: size in pixels, z: pixels per world unit, w: screen pixels per world unit
    vec4 rect;        // xy: world position of the upper left corner, zw: jitter of the rays in pixels
    uvec4 counts;     // x: number of objects, y: number of connections, z: number of triangles, w: number of light grid cells
    vec4 lightGrid;   // xy: world position of the light grid, z: size of a grid cell, w: range of the lights
    ivec4 gridSize;   // xy: number of cells of the light grid
    vec4 tissue;      // x: 1 close up, 0 zoomed out, y: visibility of the cell details, z: world size of a pixel, w: visibility of the tissue
    vec4 parameters;  // x: scale of the background color, y: bioluminescence, z: 1 for anti-aliasing within the frame, w: weight of the frame
                      // in the average with the previous frames, 1 if there is no average
    vec4 gasGrid;     // xy: world position of the first node of the gas grid, z: distance of the nodes, w: standard deviation of the blur in nodes
    ivec4 gasSize;    // xy: number of nodes of the gas grid, z: blur radius in nodes, w: number of fluid particles
    vec4 gasLight;    // x: size of the light clusters in world units, y: range of the lights, z: distance of the nodes of the irradiance grid,
                      // w: falloff distance of the lights
    ivec4 gasLightGrid;  // xy: number of light clusters, zw: number of nodes of the irradiance grid
    vec4 gasLightOrigin;  // xy: world position of the first light cluster
} frame;

layout(set = 0, binding = 1) uniform accelerationStructureEXT scene;
layout(set = 0, binding = 2, std430) readonly buffer ObjectBuffer { ObjectVertex objects[]; };
layout(set = 0, binding = 3, std430) readonly buffer LineIndexBuffer { uint lineIndices[]; };
layout(set = 0, binding = 4, std430) readonly buffer TriangleIndexBuffer { uint triangleIndices[]; };
layout(set = 0, binding = 5, std430) buffer SphereBuffer { Sphere spheres[]; };
layout(set = 0, binding = 6, std430) buffer SphereAabbBuffer { Aabb sphereAabbs[]; };
layout(set = 0, binding = 7, std430) buffer CapsuleAabbBuffer { Aabb capsuleAabbs[]; };
layout(set = 0, binding = 8, std430) buffer TriangleAabbBuffer { Aabb triangleAabbs[]; };
layout(set = 0, binding = 9, std430) buffer CellCountBuffer { uint cellCounts[]; };
layout(set = 0, binding = 10, std430) buffer CellOffsetBuffer { uint cellOffsets[]; };
layout(set = 0, binding = 11, std430) buffer CellCursorBuffer { uint cellCursors[]; };
layout(set = 0, binding = 12, std430) buffer CellEntryBuffer { uint cellEntries[]; };
layout(set = 0, binding = 13, std430) buffer AllocationBuffer { uint allocatedEntries; };
layout(set = 0, binding = 14, rgba16f) uniform image2D sceneColor;
layout(set = 0, binding = 15, rgba16f) uniform image2D antiAliased;
layout(set = 0, binding = 16) uniform sampler2D previousAntiAliased;
layout(set = 0, binding = 17) uniform sampler2D groundTexture;
layout(set = 0, binding = 18) uniform sampler2D gasSampler;
layout(set = 0, binding = 19) uniform sampler2D sceneColorSampler;
layout(set = 0, binding = 20, std430) readonly buffer FluidParticleBuffer { FluidParticle fluidParticles[]; };
layout(set = 0, binding = 21, std430) buffer GasAccumulationBuffer { uint gasAccumulation[]; };
layout(set = 0, binding = 22, std430) buffer GasIntermediateBuffer { vec4 gasIntermediate[]; };
layout(set = 0, binding = 23, rgba16f) uniform image2D gasImage;
layout(set = 0, binding = 24) uniform sampler2D energyParticleTexture;
layout(set = 0, binding = 25, std430) buffer GasLightClusterBuffer { uint gasLightClusters[]; };
layout(set = 0, binding = 26, rgba16f) uniform image2D gasIrradianceImage;
layout(set = 0, binding = 27) uniform sampler2D gasIrradianceSampler;

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

// Ratio of the visible size of a single contribution to the range of its kernel
float getContourFraction(float weight)
{
    return sqrt(1.0 - pow(DensityThreshold / weight, 1.0 / 3.0));
}

float getStrandHalfWidth()
{
    return max(mix(FarStrandHalfWidth, StrandHalfWidth, getCloseUpWeight()), MinStrandHalfWidthInPixels * getPixelSize());
}

// Zoomed out, connected cells shrink to the width of the strands, while isolated cells and the ends of chains remain dots
float getCellRadius(Sphere sphere)
{
    float farRadius = sphere.details.z > 0.5 ? max(FarDotRadius, MinDotRadiusInPixels * getPixelSize()) : getStrandHalfWidth();
    return mix(farRadius, max(sphere.centerRadius.w, farRadius), getCloseUpWeight());
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

float getStrandKernelRadius()
{
    return getStrandHalfWidth() / getContourFraction(StrandWeight);
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

//*******************************
//* Light of glowing objects
//*******************************

ivec2 getLightCell(vec2 position)
{
    return ivec2(floor((position - frame.lightGrid.xy) / frame.lightGrid.z));
}

uint getLightCellIndex(ivec2 cell)
{
    bool inGrid = cell.x >= 0 && cell.y >= 0 && cell.x < frame.gridSize.x && cell.y < frame.gridSize.y;
    return inGrid ? uint(cell.y * frame.gridSize.x + cell.x) : NoEntry;
}

bool isLight(Sphere sphere)
{
    return maxComponent(sphere.emission.rgb) * EmissionToRadiance > MinLightRadiance;
}

// The light of every glowing object in range spreads through the tissue and the surrounding medium and fades with the distance
vec3 getScatteredLight(vec2 position)
{
    vec3 result = vec3(0.0);
    float range = frame.lightGrid.w;
    ivec2 center = getLightCell(position);
    for (int i = 0; i < 9; ++i) {
        uint cellIndex = getLightCellIndex(center + ivec2(i % 3 - 1, i / 3 - 1));
        if (cellIndex == NoEntry) {
            continue;
        }
        uint offset = cellOffsets[cellIndex];
        uint count = cellCounts[cellIndex];
        for (uint j = 0u; j < count; ++j) {
            Sphere light = spheres[cellEntries[offset + j]];
            vec2 toLight = light.centerRadius.xy - position;
            float distanceSquared = dot(toLight, toLight);
            float radiusSquared = light.centerRadius.w * light.centerRadius.w;
            float window = 1.0 - smoothstep(range * 0.5, range, sqrt(distanceSquared));
            result += light.emission.rgb * EmissionToRadiance * radiusSquared / (radiusSquared + distanceSquared) * window;
        }
    }
    return result;
}

//*************************
//* Background
//*************************

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

//*******************************
//* Gas of the fluid particles
//*******************************

// The particles accumulate fixed point sums at the nodes of the gas grid, which are blurred into two layers of the gas image side by side.
// The sums per node are the number of particles, their color sum and the number of cells, which cast shadows in the gas.
const int GasImageSize = 512;
const int GasLayers = 2;
const float GasFixedPointScale = 1024.0;
const uint GasAccumulationStride = 5u;

// The cells light the gas. Their light is collected in clusters of the light cluster grid with the sums of the luminance weighted offsets
// to the cluster corner, the luminance and the color. Each cell contributes bilinearly to the four clusters around it, so that the light
// changes smoothly when cells move from one cluster to another. The offsets are shifted by one cluster to stay positive.
const uint GasLightClusterStride = 6u;
const float GasLightOffsetShift = 1.0;
const int GasIrradianceImageSize = 512;

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

// Light of the cells that reaches a position in the gas
vec3 sampleGasIrradiance(vec2 world)
{
    vec2 texel = (world - frame.gasGrid.xy) / frame.gasLight.z + 0.5;
    return sampleRegionCubic(gasIrradianceSampler, texel, vec2(0.0), vec2(frame.gasLightGrid.zw), vec2(GasIrradianceImageSize)).rgb;
}

// The energy particles and the cores of glowing fluids keep the look of the rasterized view
vec3 getEnergyParticleRadiance(vec2 world)
{
    return texture(energyParticleTexture, worldToGraphTextureCoord(world)).rgb;
}
)";
}
