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
    vec4 gasGrid;     // xy: world position of the first node of the gas grid, z: distance of the nodes, w: phase of the turbulence
    ivec4 gasSize;    // xy: number of nodes of the gas grid, z: blur radius in nodes, w: number of fluid particles
    vec4 gasSpread;   // x: standard deviation of the particle spread in nodes, y: radius of the glow around the gas in world units,
                      // z: standard deviation of the spark spread in nodes
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
            float window = smoothstep(range, range * 0.5, sqrt(distanceSquared));
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

// The left half of the image holds the gas at the nodes of the gas grid, the right half the sparks of glowing particles. The sums per node
// are fixed point numbers: number of particles, color sum and spark color sum.
const int GasImageSize = 512;
const float GasFixedPointScale = 1024.0;
const uint GasAccumulationStride = 8u;

uint getGasNodeIndex(ivec2 node)
{
    return uint(node.y * frame.gasSize.x + node.x);
}

vec2 getGasImageCoord(vec2 world, float halfOffset)
{
    vec2 node = clamp((world - frame.gasGrid.xy) / frame.gasGrid.z, vec2(0.0), vec2(frame.gasSize.xy - 1));
    return (node + vec2(halfOffset + 0.5, 0.5)) / vec2(2 * GasImageSize, GasImageSize);
}

// rgb: sum of the particle colors per area, a: number of particles per area
vec4 sampleGas(vec2 world)
{
    return texture(gasSampler, getGasImageCoord(world, 0.0));
}

// Sum of the colors of glowing particles per area
vec3 sampleSparks(vec2 world)
{
    return texture(gasSampler, getGasImageCoord(world, float(GasImageSize))).rgb;
}
)";
}
