#pragma once

#include <string_view>

namespace Shaders
{
    // Shows the creatures as living tissue under a microscope. A vertical ray through each pixel finds the groups of connected objects whose
    // range covers it: each cell alone, each connection with its two cells and each triangle with its corners. The contributions within a
    // group add up to a density. The tissue takes a smooth maximum of the groups, so that objects without a connection stay apart while
    // the transitions between overlapping groups remain invisible. The threshold of the density is the outline of the bodies. Inside, the
    // cells show wavy walls towards their neighbors and light organs that shine with their signals. On a dark background the tissue is
    // visible by its glow as in a fluorescence microscope, on a bright background by its absorption as in a brightfield microscope.
    std::string_view const RayTracingTissueCS = R"(
// Ranges of densities in which the groups blend into each other. Adjacent triangles differ by their third corner, a wider range hides the
// change of the material along their common edge.
const float GroupBlending = 0.03;
const float MaterialBlending = 0.25;

const float ThicknessGain = 1.6;

// The polygons, connections and particles of a solid add up to a density that varies with its structure. Unless the structure is visible, the
// thickness of solids saturates quickly, so that they appear as even areas with shaded edges.
const float SmoothSolidThicknessGain = 8.0;

// The glow of the tissue matches the brightness of the rasterized objects
const float Absorption = 2.2;
const float CytoplasmGlow = 1.0;
const float OutlineGlow = 1.0;
const float HaloGlow = 0.15;
const float HaloWidth = 0.08;
const float WallGlow = 0.6;
const float LightOrganRimGlow = 0.45;
const float LightOrganGlow = 1.4;
const float LightOrganHaloGlow = 0.55;
const float CoreGlow = 0.5;
const float TissueScattering = 0.4;
const float MediumScattering = 0.05;

// Zoomed out, the rasterized view adds its objects brighter, so that they glow despite their small size
const float FarGlowGain = 1.3;

// Sizes relative to the radius of the cells
const float CytoplasmGrain = 1.8;

// Each cell carries a light organ for each of the first signal channels, one in the middle and the others around it. An organ shines in
// the color of its channel while the signal is positive, see Cell::lightOrganIntensities in Engine/Impl/Kernels/Entities.cuh. Its dark disk
// absorbs light like a pigment, so that the organs remain visible on bright backgrounds. Sizes and distance are relative to the cell radius.
const vec3 LightOrganColors[LightOrgansPerCell] = vec3[](vec3(0.3, 1.0, 0.72), vec3(0.42, 0.62, 1.0), vec3(1.0, 0.42, 0.85));
const float CentralLightOrganSize = 0.19;
const float OuterLightOrganSize = 0.13;
const float OuterLightOrganDistance = 0.53;

// The walls between cells are slightly wavy, the amplitude and the frequency are given in world units
const float WallWaveAmplitude = 0.09;
const float WallWaveFrequency = 2.5;

// The tissue appears embossed as in a microscope with differential interference contrast. The relief is limited softly, so that the steep
// edges of the tissue do not turn dark on one side.
const float ReliefStrength = 0.1;
const float MaxRelief = 0.2;
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

// The share weights the contribution of the cell to its group, see getSolidSmoothing in Shaders/RayTracingCommonCS.h
void addCell(inout Tissue tissue, Sphere sphere, vec2 position, float share)
{
    if (share <= 0.0) {
        return;
    }
    vec2 offset = position - sphere.centerRadius.xy;
    float distanceSquared = dot(offset, offset);
    float kernelRadius = getCellKernelRadius(sphere);
    if (distanceSquared >= square(kernelRadius * (1.0 + CellShapeVariation))) {
        return;
    }
    float shape = mix(getCellShape(offset, sphere.details.y), 1.0, getSolidSmoothing(sphere));
    float kernelRadiusSquared = square(kernelRadius * shape);
    if (distanceSquared >= kernelRadiusSquared) {
        return;
    }
    float derivative;
    float weight = share * CellWeight * getKernel(distanceSquared / kernelRadiusSquared, derivative);
    tissue.density += weight;
    tissue.gradient += share * CellWeight * derivative * 2.0 * offset / kernelRadiusSquared;
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
    float kernelRadiusSquared = square(getStrandKernelRadius(getStrandSolidSmoothing(a, b)));
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
            addCell(group, sphere, position, 1.0);
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
            addCell(group, a, position, 1.0 - getSolidSmoothing(a));
            addCell(group, b, position, 1.0 - getSolidSmoothing(b));
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
            addCell(group, a, position, 1.0 - getSolidSmoothing(a));
            addCell(group, b, position, 1.0 - getSolidSmoothing(b));
            addCell(group, c, position, 1.0 - getSolidSmoothing(c));
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
    float lightOrgan;
    float lightOrganRim;
    vec3 lightOrganEmission;
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

// See packLightOrganIntensities in Shaders/RayTracingPrepareCS.h
bool hasLightOrgans(Sphere cell)
{
    return cell.details.x >= 0.0;
}

float getLightOrganIntensity(Sphere cell, int index)
{
    return float((uint(cell.details.x) >> (8 * index)) & 255u) / 255.0;
}

// The arrangement of the organs turns with the seed of the cell. Their light grows faster than linear with the signals, so that weak signals
// remain dim.
void addLightOrgans(inout CellDetails details, Sphere cell, vec2 local, float visibility)
{
    float radius = cell.centerRadius.w;
    float seed = cell.details.y;
    float pixelSize = getPixelSize();
    float edge = max(0.05 * radius, pixelSize);
    for (int i = 0; i < LightOrgansPerCell; ++i) {
        float size = radius * (i == 0 ? CentralLightOrganSize : OuterLightOrganSize);
        float angle = 6.2831853 * (seed + float(i - 1) / float(max(LightOrgansPerCell - 1, 1)));
        float distanceToCenter = i == 0 ? 0.0 : radius * (OuterLightOrganDistance + 0.1 * hash(vec2(seed, float(i))));
        float distance = length(local - distanceToCenter * vec2(cos(angle), sin(angle)));

        // Organs smaller than a pixel would only appear as single bright pixels
        float organVisibility = smoothstep(0.8, 2.0, size / pixelSize) * visibility;
        details.lightOrgan = max(details.lightOrgan, (1.0 - smoothstep(size - edge, size, distance)) * organVisibility);
        details.lightOrganRim += exp(-square((distance - size) / max(0.02, 0.7 * pixelSize))) * organVisibility;

        float intensity = pow(getLightOrganIntensity(cell, i), 1.5);
        float core = 1.0 - smoothstep(0.5 * size - edge, 0.5 * size, distance);
        float halo = exp(-square(distance / (2.0 * size)));
        details.lightOrganEmission += LightOrganColors[i] * intensity * (LightOrganGlow * core + LightOrganHaloGlow * halo) * organVisibility;
    }
}

// The cell at a position is the nearest one, the walls lie where the nearest and the second nearest cell are equally far away
CellDetails getCellDetails(Neighborhood neighborhood, vec2 position)
{
    CellDetails result = CellDetails(0.0, 0.0, 0.0, vec3(0.0), 0.0);
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
        result.wall *= getSolidDetailVisibility();
        return result;
    }
    vec2 local = position - cell.centerRadius.xy;
    result.grain = (valueNoise(local / cell.centerRadius.w * CytoplasmGrain + cell.details.y * 29.0) - 0.5) * visibility;
    if (hasLightOrgans(cell)) {
        addLightOrgans(result, cell, local, visibility);
    }
    return result;
}

// The material of the gathered tissue is normalized, see getBlendedTissue
vec3 getTissueColor(Tissue tissue)
{
    return getSaturatedHue(tissue.hue);
}

// Light scattered at the edge of the tissue surrounds it with a faint halo as in a dark field microscope
vec3 getHalo(Tissue tissue)
{
    float distance = max(DensityThreshold - tissue.density, 0.0) / max(length(tissue.gradient), 1e-4);
    return getTissueColor(tissue) * tissue.brightness * HaloGlow * getCloseUpWeight() * exp(-distance / HaloWidth) * (1.0 - tissue.solid);
}

// The tissue and the solids scatter the light of the radiation sources in their colors, see Shaders/RayTracingGasIrradianceCS.h
const float TissueSourceLight = 1.6;
const float SolidSourceLight = 1.2;

// Zoomed out, the structures are only a few pixels wide and appear as dense lines without outline
vec3 shadeTissue(Tissue tissue, Neighborhood neighborhood, vec2 position, vec3 medium, vec3 scattered, vec3 sourceLight)
{
    float closeUpWeight = getCloseUpWeight();
    float excess = max(tissue.density - DensityThreshold, 0.0);
    float edgeDistance = excess / max(length(tissue.gradient), 1e-4);
    float outline = exp(-square(edgeDistance / max(0.05, 0.8 * getPixelSize()))) * closeUpWeight;

    float totalExcess = max(tissue.totalDensity - DensityThreshold, 0.0);
    float thicknessGain = mix(ThicknessGain, SmoothSolidThicknessGain, tissue.solid * (1.0 - getSolidDetailVisibility()));
    float thickness = mix(1.0, 1.0 - exp(-totalExcess * thicknessGain), closeUpWeight);
    vec2 thicknessGradient = closeUpWeight * thicknessGain * exp(-totalExcess * thicknessGain) * tissue.totalGradient;

    vec3 color = getTissueColor(tissue);
    CellDetails details = getCellDetails(neighborhood, position);
    float cytoplasm = thickness * (1.0 + 0.35 * details.grain) * (1.0 - 0.55 * details.lightOrgan);

    float opticalDensity = cytoplasm + outline * 0.8 + details.wall * 0.6 + details.lightOrgan * 0.5;
    vec3 transmittance = exp(-Absorption * opticalDensity * getAbsorbedShare(color));
    vec3 fluorescence = color * tissue.brightness
        * (CytoplasmGlow * cytoplasm + OutlineGlow * outline + WallGlow * details.wall + LightOrganRimGlow * details.lightOrganRim);
    vec3 glow = tissue.emission * CoreGlow * min(tissue.bulge, 1.0) + details.lightOrganEmission
        + (scattered * TissueScattering + sourceLight * TissueSourceLight) * thickness * color;
    float relief = 1.0 + MaxRelief * tanh(ReliefStrength * dot(thicknessGradient, ReliefDirection) / MaxRelief);
    vec3 result = (medium * transmittance + (fluorescence + glow) * mix(FarGlowGain, 1.0, closeUpWeight)) * relief;

    // Solids are dense grains without the inner structure of cells
    vec3 solidColor =
        (medium * exp(-6.0 * opticalDensity) + color * tissue.brightness * (0.3 + 0.5 * thickness) + sourceLight * SolidSourceLight * color) * relief;
    return mix(result, solidColor, tissue.solid);
}

// As in the rasterized view, the background is added after the bloom and the tone mapping, so the light in front of it is passed on alone.
// Where the tissue absorbs the background, the darkening is converted, so that adding the background yields the tone mapped absorption. The
// bloom scales the scene by the inverse of the background scale.
vec3 getForeground(vec3 color, vec3 background)
{
    vec3 foreground = color - background;
    float bloomScale = 1.0 / frame.parameters.x;
    vec3 scaledBackground = background * bloomScale;
    vec3 darkening = toneMapping(scaledBackground) - toneMapping(max(scaledBackground + foreground * bloomScale, vec3(0.0)));
    vec3 absorption = -inverseToneMapping(darkening) / bloomScale;
    return mix(absorption, foreground, step(vec3(0.0), foreground));
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
    vec3 background = getBackgroundRadiance(position);
    vec3 sourceLight = isSourceLightActive() ? getSourceLight(position) : vec3(0.0);
    vec3 medium = shadeGas(position, background, sourceLight) + getEnergyParticleRadiance(position);
    vec3 color = medium;
    if (tissueVisibility > 0.0) {
        vec3 lightedMedium = medium + scattered * MediumScattering;
        color = lightedMedium;
        Neighborhood neighborhood;
        Tissue tissue = gatherTissue(position, neighborhood);
        if (tissue.weight > 0.0) {
            float coverage = getCoverage(tissue);
            color = mix(lightedMedium + getHalo(tissue), shadeTissue(tissue, neighborhood, position, medium, scattered, sourceLight), coverage);
        }
        color = mix(medium, color, tissueVisibility);
    }

    // The rasterized objects scatter the light of the radiation sources as well, its strength is passed on to them in the alpha channel,
    // see Shaders/RayTracingMergeFS.h
    imageStore(sceneColor, pixel, vec4(getForeground(color, background), luminance(sourceLight) * TissueSourceLight));
}
)";
}
