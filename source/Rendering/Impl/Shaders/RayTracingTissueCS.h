#pragma once

#include <string_view>

namespace Shaders
{
    // Shows the creatures as living tissue under a microscope. A vertical ray through each pixel finds the groups of connected objects whose
    // range covers it: each cell alone, each connection with its two cells and each triangle with its corners. The contributions within a
    // group add up to a density. The tissue takes a smooth maximum of the groups, so that objects without a connection stay apart while
    // the transitions between overlapping groups remain invisible. The threshold of the density is the outline of the bodies. Inside, the
    // cells show wavy walls towards their neighbors, a nucleus and vesicles. On a dark background the tissue is visible by its glow as in
    // a fluorescence microscope, on a bright background by its absorption as in a brightfield microscope.
    std::string_view const RayTracingTissueCS = R"(
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

// Ranges of densities in which the groups blend into each other. Adjacent triangles differ by their third corner, a wider range hides the
// change of the material along their common edge.
const float GroupBlending = 0.03;
const float MaterialBlending = 0.25;

const float ThicknessGain = 1.6;

// The glow of the tissue matches the brightness of the rasterized objects
const float Saturation = 0.9;
const float Absorption = 2.2;
const float CytoplasmGlow = 1.0;
const float OutlineGlow = 1.0;
const float HaloGlow = 0.15;
const float HaloWidth = 0.08;
const float WallGlow = 0.6;
const float NucleusRimGlow = 0.25;
const float NucleolusGlow = 0.5;
const float VesicleGlow = 0.6;
const float CoreGlow = 0.5;
const float TissueScattering = 0.4;
const float MediumScattering = 0.05;

// Zoomed out, the rasterized view adds its objects brighter, so that they glow despite their small size
const float FarGlowGain = 1.3;

// Sizes relative to the radius of the cells
const float NucleusSize = 0.4;
const float VesicleSpacing = 0.62;
const float CytoplasmGrain = 1.8;

// The walls between cells are slightly wavy, the amplitude and the frequency are given in world units
const float WallWaveAmplitude = 0.09;
const float WallWaveFrequency = 2.5;

// The tissue appears embossed as in a microscope with differential interference contrast
const float ReliefStrength = 0.1;
const vec2 ReliefDirection = vec2(-0.7071, -0.7071);

// Density and material of a group of connected objects
struct Tissue
{
    float density;
    vec2 gradient;
    // Sum of all contributions without groups and its gradient. Its transitions are smooth everywhere, which is required for the thickness
    // and the relief inside the bodies.
    float totalDensity;
    vec2 totalGradient;
    float bulge;
    vec3 hue;
    float weight;
    vec3 emission;
    float brightness;
    float solid;
};

Tissue createTissue()
{
    return Tissue(0.0, vec2(0.0), 0.0, vec2(0.0), 0.0, vec3(0.0), 0.0, vec3(0.0), 0.0, 0.0);
}

// Smooth maximum of the group densities as the logarithm of a sum of exponentials. The gradient of the density is averaged with the same
// weights, the materials with weights of the wider range. All weights are relative to the highest density found so far, which keeps the
// exponentials in range.
struct TissueBlend
{
    float maxDensity;
    float weightSum;
    float materialWeightSum;
    Tissue weightedSum;
};

TissueBlend createTissueBlend()
{
    return TissueBlend(0.0, 0.0, 0.0, createTissue());
}

void scaleMaterial(inout Tissue sum, float factor)
{
    sum.bulge *= factor;
    sum.hue *= factor;
    sum.emission *= factor;
    sum.brightness *= factor;
    sum.solid *= factor;
}

void addGroup(inout TissueBlend blend, Tissue group)
{
    if (group.weight <= 0.0) {
        return;
    }
    if (group.density > blend.maxDensity) {
        float densityDifference = blend.maxDensity - group.density;
        float factor = exp(densityDifference / GroupBlending);
        float materialFactor = exp(densityDifference / MaterialBlending);
        blend.weightSum *= factor;
        blend.weightedSum.gradient *= factor;
        blend.materialWeightSum *= materialFactor;
        scaleMaterial(blend.weightedSum, materialFactor);
        blend.maxDensity = group.density;
    }
    float densityDifference = group.density - blend.maxDensity;
    float weight = exp(densityDifference / GroupBlending);
    float materialWeight = exp(densityDifference / MaterialBlending);
    blend.weightSum += weight;
    blend.weightedSum.gradient += weight * group.gradient;
    blend.materialWeightSum += materialWeight;
    blend.weightedSum.bulge += materialWeight * group.bulge;
    materialWeight /= group.weight;
    blend.weightedSum.hue += materialWeight * group.hue;
    blend.weightedSum.emission += materialWeight * group.emission;
    blend.weightedSum.brightness += materialWeight * group.brightness;
    blend.weightedSum.solid += materialWeight * group.solid;
}

// The material of the result is normalized, i.e. its weight is 1
Tissue getBlendedTissue(TissueBlend blend)
{
    if (blend.weightSum <= 0.0) {
        return createTissue();
    }
    Tissue result = blend.weightedSum;
    scaleMaterial(result, 1.0 / blend.materialWeightSum);
    result.gradient /= blend.weightSum;
    result.density = blend.maxDensity + GroupBlending * log(blend.weightSum);
    result.weight = 1.0;
    return result;
}

// The two cells nearest to a position, the walls lie where both are equally far away
struct Neighborhood
{
    uint nearestSphere;
    float nearestDistanceSquared;
    uint secondNearestSphere;
    float secondNearestDistanceSquared;
};

Neighborhood createNeighborhood()
{
    return Neighborhood(NoEntry, 1e20, NoEntry, 1e20);
}

float square(float value)
{
    return value * value;
}

// Smooth kernel that vanishes at its range, the argument is the squared distance relative to the range
float getKernel(float squaredRatio, out float derivative)
{
    float remainder = max(1.0 - squaredRatio, 0.0);
    derivative = -3.0 * remainder * remainder;
    return remainder * remainder * remainder;
}

void addMaterial(inout Tissue tissue, Sphere sphere, float weight)
{
    tissue.hue += sphere.hueMaterial.rgb * weight;
    tissue.weight += weight;
    tissue.emission += sphere.emission.rgb * weight;
    tissue.brightness += sphere.emission.w * weight;
    tissue.solid += sphere.hueMaterial.w == Material_Solid ? weight : 0.0;
}

// Living cells are not perfectly round, their outline varies with the direction
float getCellShape(vec2 offset, float seed)
{
    float angle = atan(offset.y, offset.x);
    return 1.0 + CellShapeVariation * (0.6 * sin(2.0 * angle + seed * 47.0) + 0.4 * sin(3.0 * angle + seed * 91.0));
}

void addCell(inout Tissue tissue, Sphere sphere, vec2 position)
{
    vec2 offset = position - sphere.centerRadius.xy;
    float distanceSquared = dot(offset, offset);
    float kernelRadius = getCellKernelRadius(sphere);
    if (distanceSquared >= square(kernelRadius * (1.0 + CellShapeVariation))) {
        return;
    }
    float kernelRadiusSquared = square(kernelRadius * getCellShape(offset, sphere.details.y));
    if (distanceSquared >= kernelRadiusSquared) {
        return;
    }
    float derivative;
    float weight = CellWeight * getKernel(distanceSquared / kernelRadiusSquared, derivative);
    tissue.density += weight;
    tissue.gradient += CellWeight * derivative * 2.0 * offset / kernelRadiusSquared;
    tissue.bulge += weight;
    addMaterial(tissue, sphere, weight);
}

void addToNeighborhood(inout Neighborhood neighborhood, uint index, Sphere sphere, vec2 position)
{
    vec2 offset = position - sphere.centerRadius.xy;
    float distanceSquared = dot(offset, offset);
    if (distanceSquared < neighborhood.nearestDistanceSquared) {
        neighborhood.secondNearestSphere = neighborhood.nearestSphere;
        neighborhood.secondNearestDistanceSquared = neighborhood.nearestDistanceSquared;
        neighborhood.nearestSphere = index;
        neighborhood.nearestDistanceSquared = distanceSquared;
    } else if (distanceSquared < neighborhood.secondNearestDistanceSquared) {
        neighborhood.secondNearestSphere = index;
        neighborhood.secondNearestDistanceSquared = distanceSquared;
    }
}

void addStrand(inout Tissue tissue, Sphere a, Sphere b, vec2 position)
{
    vec2 axis = b.centerRadius.xy - a.centerRadius.xy;
    float axisParameter = clamp(dot(position - a.centerRadius.xy, axis) / max(dot(axis, axis), 1e-8), 0.0, 1.0);
    vec2 offset = position - (a.centerRadius.xy + axis * axisParameter);
    float distanceSquared = dot(offset, offset);
    float kernelRadiusSquared = square(getStrandKernelRadius());
    if (distanceSquared >= kernelRadiusSquared) {
        return;
    }
    float derivative;
    float weight = StrandWeight * getKernel(distanceSquared / kernelRadiusSquared, derivative);
    tissue.density += weight;
    tissue.gradient += StrandWeight * derivative * 2.0 * offset / kernelRadiusSquared;
    addMaterial(tissue, a, weight * (1.0 - axisParameter));
    addMaterial(tissue, b, weight * axisParameter);
}

float cross2(vec2 a, vec2 b)
{
    return a.x * b.y - a.y * b.x;
}

// Signed distance to a triangle and its gradient, see https://iquilezles.org/articles/distgradfunctions2d
vec3 getTriangleDistance(vec2 position, vec2 cornerA, vec2 cornerB, vec2 cornerC)
{
    float orientation = cross2(cornerA - cornerC, cornerB - cornerA);
    vec2 edgeA = cornerB - cornerA;
    vec2 edgeB = cornerC - cornerB;
    vec2 edgeC = cornerA - cornerC;
    vec2 toA = position - cornerA;
    vec2 toB = position - cornerB;
    vec2 toC = position - cornerC;
    vec2 offsetA = toA - edgeA * clamp(dot(toA, edgeA) / max(dot(edgeA, edgeA), 1e-12), 0.0, 1.0);
    vec2 offsetB = toB - edgeB * clamp(dot(toB, edgeB) / max(dot(edgeB, edgeB), 1e-12), 0.0, 1.0);
    vec2 offsetC = toC - edgeC * clamp(dot(toC, edgeC) / max(dot(edgeC, edgeC), 1e-12), 0.0, 1.0);
    vec3 closest = vec3(dot(offsetA, offsetA), offsetA);
    if (dot(offsetB, offsetB) < closest.x) {
        closest = vec3(dot(offsetB, offsetB), offsetB);
    }
    if (dot(offsetC, offsetC) < closest.x) {
        closest = vec3(dot(offsetC, offsetC), offsetC);
    }
    float side = max(max(orientation * cross2(toA, edgeA), orientation * cross2(toB, edgeB)), orientation * cross2(toC, edgeC));
    float signedDistance = sqrt(closest.x) * (side > 0.0 ? 1.0 : -1.0);
    return vec3(signedDistance, closest.yz / (abs(signedDistance) > 1e-6 ? signedDistance : 1e-6));
}

vec3 getBarycentrics(vec2 position, vec2 cornerA, vec2 cornerB, vec2 cornerC)
{
    vec2 edgeB = cornerB - cornerA;
    vec2 edgeC = cornerC - cornerA;
    vec2 toPosition = position - cornerA;
    float dotBB = dot(edgeB, edgeB);
    float dotBC = dot(edgeB, edgeC);
    float dotCC = dot(edgeC, edgeC);
    float denominator = dotBB * dotCC - dotBC * dotBC;
    if (abs(denominator) < 1e-10) {
        return vec3(1.0 / 3.0);
    }
    float weightB = (dotCC * dot(toPosition, edgeB) - dotBC * dot(toPosition, edgeC)) / denominator;
    float weightC = (dotBB * dot(toPosition, edgeC) - dotBC * dot(toPosition, edgeB)) / denominator;
    vec3 result = max(vec3(1.0 - weightB - weightC, weightB, weightC), vec3(0.0));
    return result / max(result.x + result.y + result.z, 1e-6);
}

void addFill(inout Tissue tissue, Sphere a, Sphere b, Sphere c, vec2 position)
{
    vec3 distanceGradient = getTriangleDistance(position, a.centerRadius.xy, b.centerRadius.xy, c.centerRadius.xy);
    float falloff = getFillFalloff();
    float outside = max(distanceGradient.x, 0.0);
    if (outside >= falloff) {
        return;
    }
    float derivative;
    float weight = FillWeight * getKernel(square(outside / falloff), derivative);
    tissue.density += weight;
    tissue.gradient += FillWeight * derivative * 2.0 * outside / (falloff * falloff) * distanceGradient.yz;
    vec3 barycentrics = getBarycentrics(position, a.centerRadius.xy, b.centerRadius.xy, c.centerRadius.xy);
    addMaterial(tissue, a, weight * barycentrics.x);
    addMaterial(tissue, b, weight * barycentrics.y);
    addMaterial(tissue, c, weight * barycentrics.z);
}
)"
                                                R"(
// The bounding boxes cover the range of each group, hence the ray finds all groups that reach the pixel. Each group reaches zero density
// at the border of its bounding box, so that the tissue stays continuous.
Tissue gatherTissue(vec2 position, out Neighborhood neighborhood)
{
    TissueBlend blend = createTissueBlend();
    neighborhood = createNeighborhood();
    float totalDensity = 0.0;
    vec2 totalGradient = vec2(0.0);
    rayQueryEXT query;
    rayQueryInitializeEXT(query, scene, gl_RayFlagsNoneEXT, 0xffu, vec3(position, CameraHeight), 0.0, vec3(0.0, 0.0, -1.0), 2.0 * CameraHeight);
    while (rayQueryProceedEXT(query)) {
        uint kind = uint(rayQueryGetIntersectionGeometryIndexEXT(query, false)) + 1u;
        uint primitive = uint(rayQueryGetIntersectionPrimitiveIndexEXT(query, false));

        // Each group starts with its own contribution, which enters the total, the cells of connections and triangles follow
        Tissue group = createTissue();
        if (kind == Kind_Sphere) {
            Sphere sphere = spheres[primitive];
            addCell(group, sphere, position);
            totalDensity += group.density;
            totalGradient += group.gradient;
            if (group.weight > 0.0) {
                addToNeighborhood(neighborhood, primitive, sphere, position);
            }
        } else if (kind == Kind_Capsule) {
            uint indexA;
            uint indexB;
            getCapsuleSpheres(primitive, indexA, indexB);
            Sphere a = spheres[indexA];
            Sphere b = spheres[indexB];
            addStrand(group, a, b, position);
            totalDensity += group.density;
            totalGradient += group.gradient;
            addCell(group, a, position);
            addCell(group, b, position);
        } else if (kind == Kind_Triangle) {
            uint indexA;
            uint indexB;
            uint indexC;
            getTriangleSpheres(primitive, indexA, indexB, indexC);
            Sphere a = spheres[indexA];
            Sphere b = spheres[indexB];
            Sphere c = spheres[indexC];
            addFill(group, a, b, c, position);
            totalDensity += group.density;
            totalGradient += group.gradient;
            addCell(group, a, position);
            addCell(group, b, position);
            addCell(group, c, position);
        }
        addGroup(blend, group);
    }
    Tissue result = getBlendedTissue(blend);
    result.totalDensity = totalDensity;
    result.totalGradient = totalGradient;
    return result;
}

float getCoverage(Tissue tissue)
{
    return clamp((tissue.density - DensityThreshold) / max(length(tissue.gradient) * getPixelSize(), 1e-5) + 0.5, 0.0, 1.0);
}

struct CellDetails
{
    float wall;
    float nucleus;
    float nucleusRim;
    float nucleolus;
    float vesicle;
    float grain;
};

// The distance is measured from the side of the nearest cell. The waves stay with the cells, since they derive from the seeds of both.
float getWallDistance(Sphere cell, Sphere neighbor, Neighborhood neighborhood, vec2 position)
{
    vec2 axis = neighbor.centerRadius.xy - cell.centerRadius.xy;
    float centerDistance = max(length(axis), 1e-4);
    float wallDistance = (neighborhood.secondNearestDistanceSquared - neighborhood.nearestDistanceSquared) / (2.0 * centerDistance);

    // Both cells have to see the same wall, hence the waves are defined in an orientation that does not depend on which one is nearer
    float orientation = cell.details.y < neighbor.details.y ? 1.0 : -1.0;
    vec2 direction = axis / centerDistance * orientation;
    float along = dot(position - (cell.centerRadius.xy + neighbor.centerRadius.xy) * 0.5, vec2(-direction.y, direction.x));
    float pairSeed = cell.details.y + neighbor.details.y;
    float wave = (valueNoise(vec2(along * WallWaveFrequency + pairSeed * 53.0, pairSeed * 31.0)) - 0.5) * WallWaveAmplitude;
    return abs(wallDistance * orientation - wave);
}

// Small vesicles of different sizes lie scattered in the cytoplasm
float getVesicles(vec2 local, float radius, float seed)
{
    float spacing = radius * VesicleSpacing;
    vec2 grid = local / spacing + seed * 17.0;
    vec2 base = floor(grid);
    float pixelSize = getPixelSize() / spacing;
    float edge = max(0.06, pixelSize);
    float result = 0.0;
    for (int i = 0; i < 9; ++i) {
        vec2 gridCell = base + vec2(i % 3 - 1, i / 3 - 1);
        if (hash(gridCell + 0.37) < 0.55) {
            continue;
        }
        vec2 center = gridCell + 0.2 + 0.6 * vec2(hash(gridCell), hash(gridCell + 7.1));
        float size = mix(0.1, 0.24, hash(gridCell + 3.3));

        // Vesicles smaller than a pixel would only appear as single bright pixels
        float visibility = smoothstep(0.8, 2.0, size / pixelSize);
        result = max(result, (1.0 - smoothstep(size - edge, size, length(grid - center))) * visibility);
    }
    return result;
}

// The cell at a position is the nearest one, the walls lie where the nearest and the second nearest cell are equally far away
CellDetails getCellDetails(Neighborhood neighborhood, vec2 position)
{
    CellDetails result = CellDetails(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    float visibility = frame.tissue.y;
    if (visibility <= 0.0 || neighborhood.nearestSphere == NoEntry) {
        return result;
    }
    float pixelSize = getPixelSize();
    Sphere cell = spheres[neighborhood.nearestSphere];
    if (neighborhood.secondNearestSphere != NoEntry) {
        float wallDistance = getWallDistance(cell, spheres[neighborhood.secondNearestSphere], neighborhood, position);
        result.wall = exp(-square(wallDistance / max(0.035, 0.7 * pixelSize))) * visibility;
    }
    if (cell.hueMaterial.w == Material_Solid) {
        return result;
    }
    float seed = cell.details.y;
    float radius = cell.centerRadius.w;
    vec2 local = position - cell.centerRadius.xy;

    vec2 nucleusOffset = (vec2(hash(vec2(seed, 0.13)), hash(vec2(seed, 0.71))) - 0.5) * 0.37 * radius;
    vec2 toNucleus = local - nucleusOffset;
    float angle = atan(toNucleus.y, toNucleus.x);
    float shape = 1.0 + 0.08 * sin(3.0 * angle + seed * 40.0) + 0.05 * sin(5.0 * angle + seed * 73.0);
    float nucleusRadius = radius * (NucleusSize + 0.08 * hash(vec2(seed, 0.37))) * shape;
    float nucleusDistance = length(toNucleus);
    float nucleusEdge = max(0.05 * radius, pixelSize);
    result.nucleus = (1.0 - smoothstep(nucleusRadius - nucleusEdge, nucleusRadius, nucleusDistance)) * visibility;
    result.nucleusRim = exp(-square((nucleusDistance - nucleusRadius) / max(0.03, 0.7 * pixelSize))) * visibility;

    vec2 nucleolusOffset = (vec2(hash(vec2(seed, 0.53)), hash(vec2(seed, 0.91))) - 0.5) * 0.7 * nucleusRadius;
    float nucleolusRadius = nucleusRadius * 0.3;
    result.nucleolus = (1.0 - smoothstep(nucleolusRadius - nucleusEdge, nucleolusRadius, length(toNucleus - nucleolusOffset))) * result.nucleus
        * smoothstep(0.8, 2.0, nucleolusRadius / pixelSize);

    result.vesicle = getVesicles(local, radius, seed) * (1.0 - result.nucleus) * visibility;
    result.grain = (valueNoise(local / radius * CytoplasmGrain + seed * 29.0) - 0.5) * visibility;
    return result;
}

vec3 getTissueColor(Tissue tissue)
{
    vec3 hue = tissue.hue / tissue.weight;
    return mix(vec3(luminance(hue)), hue, Saturation);
}

// Light scattered at the edge of the tissue surrounds it with a faint halo as in a dark field microscope
vec3 getHalo(Tissue tissue)
{
    float distance = max(DensityThreshold - tissue.density, 0.0) / max(length(tissue.gradient), 1e-4);
    float brightness = tissue.brightness / tissue.weight;
    float solid = tissue.solid / tissue.weight;
    return getTissueColor(tissue) * brightness * HaloGlow * getCloseUpWeight() * exp(-distance / HaloWidth) * (1.0 - solid);
}

// Zoomed out, the structures are only a few pixels wide and appear as dense lines without outline
vec3 shadeTissue(Tissue tissue, Neighborhood neighborhood, vec2 position, vec3 medium, vec3 scattered)
{
    float closeUpWeight = getCloseUpWeight();
    float excess = max(tissue.density - DensityThreshold, 0.0);
    float edgeDistance = excess / max(length(tissue.gradient), 1e-4);
    float outline = exp(-square(edgeDistance / max(0.05, 0.8 * getPixelSize()))) * closeUpWeight;

    float totalExcess = max(tissue.totalDensity - DensityThreshold, 0.0);
    float thickness = mix(1.0, 1.0 - exp(-totalExcess * ThicknessGain), closeUpWeight);
    vec2 thicknessGradient = closeUpWeight * ThicknessGain * exp(-totalExcess * ThicknessGain) * tissue.totalGradient;

    vec3 color = getTissueColor(tissue);
    float brightness = tissue.brightness / tissue.weight;
    vec3 emission = tissue.emission / tissue.weight;
    float solid = tissue.solid / tissue.weight;
    CellDetails details = getCellDetails(neighborhood, position);
    float cytoplasm = thickness * (1.0 + 0.35 * details.grain) * (1.0 - 0.55 * details.nucleus);

    float opticalDensity =
        cytoplasm + outline * 0.8 + details.wall * 0.6 + details.nucleus * 0.45 + details.nucleolus * 0.5 + details.vesicle * 0.3;
    vec3 transmittance = exp(-Absorption * opticalDensity * (vec3(1.0) - color * 0.85));
    vec3 fluorescence = color * brightness
        * (CytoplasmGlow * cytoplasm + OutlineGlow * outline + WallGlow * details.wall + NucleusRimGlow * details.nucleusRim
           + NucleolusGlow * details.nucleolus + VesicleGlow * details.vesicle);
    vec3 glow = emission * CoreGlow * min(tissue.bulge, 1.0) + scattered * TissueScattering * thickness * color;
    float relief = max(1.0 + ReliefStrength * dot(thicknessGradient, ReliefDirection), 0.2);
    vec3 result = (medium * transmittance + (fluorescence + glow) * mix(FarGlowGain, 1.0, closeUpWeight)) * relief;

    // Solids are dense grains without the inner structure of cells
    vec3 solidColor = (medium * exp(-6.0 * opticalDensity) + color * brightness * (0.3 + 0.5 * thickness)) * relief;
    return mix(result, solidColor, solid);
}

void main()
{
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    if (!isInView(pixel)) {
        return;
    }
    vec2 position = pixelToWorld(vec2(pixel) + 0.5 + frame.rect.zw);

    // Zoomed out, the rasterized objects replace the tissue, whose scene and lights are not built then
    float tissueVisibility = frame.tissue.w;
    vec3 scattered = tissueVisibility > 0.0 ? getScatteredLight(position) : vec3(0.0);
    vec3 medium = shadeGas(position, getBackgroundRadiance(position)) + getEnergyParticleRadiance(position);
    vec3 color = medium;
    if (tissueVisibility > 0.0) {
        vec3 lightedMedium = medium + scattered * MediumScattering;
        color = lightedMedium;
        Neighborhood neighborhood;
        Tissue tissue = gatherTissue(position, neighborhood);
        if (tissue.weight > 0.0) {
            float coverage = getCoverage(tissue);
            color = mix(lightedMedium + getHalo(tissue), shadeTissue(tissue, neighborhood, position, medium, scattered), coverage);
        }
        color = mix(medium, color, tissueVisibility);
    }
    imageStore(sceneColor, pixel, vec4(color, 1.0));
}
)";
}
