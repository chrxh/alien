#pragma once

#include <string_view>

namespace Shaders
{
    // Shows the creatures as living tissue under a microscope. A vertical ray through each pixel finds the groups of connected objects whose
    // range covers it: each cell alone, each connection with its two cells and each triangle with its corners. The contributions within a
    // group add up to a density. The tissue takes a smooth maximum of the groups, so that objects without a connection stay apart while
    // the transitions between overlapping groups remain invisible. The threshold of the density is the outline of the bodies. Inside, the
    // cells show wavy walls towards their neighbors and small organelles. Each cell is a neuron with a body and dendrites towards its
    // neighbors, which glow with its activity and pass on a pulse of light when it fires. On a dark background the tissue is visible by its
    // glow as in a fluorescence microscope, on a bright background by its absorption as in a brightfield microscope.
    std::string_view const RayTracingTissueCS = R"(
const float TwoPi = 6.2831853;

// Ranges of densities in which the groups blend into each other. Adjacent triangles differ by their third corner, a wider range hides the
// change of the material along their common edge.
const float GroupBlending = 0.03;
const float MaterialBlending = 0.25;

// The polygons of the tissue dominate the material and the cells at a position over the other groups, so that a strand lying on a body
// without a connection to it hides below the body instead of replacing a part of it
const float PolygonDominance = 1.5;

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

// Zoomed out, the rasterized view adds its objects brighter, so that they glow despite their small size
const float FarGlowGain = 1.3;

// Zoomed out, the cells show their own colors instead of the blended material, so that they stay apart without their walls and inner
// details
const float CellColorSharpness = 0.7;

// Sizes relative to the radius of the cells
const float CytoplasmGrain = 1.8;

// Each cell contains a few small organelles. They do not glow, so they appear as dark spots in the fluorescent cytoplasm and absorb light
// like a pigment on bright backgrounds. Sizes and distances are relative to the cell radius.
const int OrganellesPerCell = 4;
const float MinOrganelleSize = 0.09;
const float MaxOrganelleSize = 0.18;
const float MaxOrganelleDistance = 0.9;
const float OrganelleEdge = 0.45;
const float OrganelleShade = 0.4;
const float OrganelleDensity = 0.4;

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
    vec3 hue;
    float weight;
    float brightness;
    float solid;
};

Tissue createTissue()
{
    return Tissue(0.0, vec2(0.0), 0.0, vec2(0.0), vec3(0.0), 0.0, 0.0, 0.0);
}

// Smooth maximum of the group densities as the logarithm of a sum of exponentials. The gradient of the density is averaged with the same
// weights, the materials with weights of the wider range applied to the dominance of the groups, i.e. their density raised for polygons. A
// group fades out of the material with its density, so that the material does not change abruptly where the range of the group ends. All
// weights are relative to the highest density or dominance found so far, which keeps the exponentials in range.
struct TissueBlend
{
    float maxDensity;
    float maxDominance;
    float weightSum;
    float materialWeightSum;
    Tissue weightedSum;
};

TissueBlend createTissueBlend()
{
    return TissueBlend(0.0, 0.0, 0.0, 0.0, createTissue());
}

void scaleMaterial(inout Tissue sum, float factor)
{
    sum.hue *= factor;
    sum.brightness *= factor;
    sum.solid *= factor;
}

void addGroup(inout TissueBlend blend, Tissue group, float dominance)
{
    if (group.weight <= 0.0) {
        return;
    }
    if (group.density > blend.maxDensity) {
        float factor = exp((blend.maxDensity - group.density) / GroupBlending);
        blend.weightSum *= factor;
        blend.weightedSum.gradient *= factor;
        blend.maxDensity = group.density;
    }
    if (dominance > blend.maxDominance) {
        float materialFactor = exp((blend.maxDominance - dominance) / MaterialBlending);
        blend.materialWeightSum *= materialFactor;
        scaleMaterial(blend.weightedSum, materialFactor);
        blend.maxDominance = dominance;
    }
    float weight = exp((group.density - blend.maxDensity) / GroupBlending);
    float materialWeight = exp((dominance - blend.maxDominance) / MaterialBlending) * clamp(group.density / DensityThreshold, 0.0, 1.0);
    blend.weightSum += weight;
    blend.weightedSum.gradient += weight * group.gradient;
    blend.materialWeightSum += materialWeight;
    materialWeight /= group.weight;
    blend.weightedSum.hue += materialWeight * group.hue;
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
    scaleMaterial(result, 1.0 / max(blend.materialWeightSum, 1e-30));
    result.gradient /= blend.weightSum;
    result.density = blend.maxDensity + GroupBlending * log(blend.weightSum);
    result.weight = 1.0;
    return result;
}

// The cells nearest to a position in the order of their rank, i.e. their squared distance reduced by the dominance of their strongest group
// at the position, see PolygonDominance. The walls lie where the nearest and another cell have the same rank, three cells meet where the
// walls join. A cell of a body passing over another one thus does not draw walls into it, while the cells of a body share their groups and
// keep their walls in the middle. A cell counts within the largest range of its kernel, independent of the variation of its outline, so
// that the nearest cells do not change at the edges of the kernels. Connections and polygons contribute their cells wherever they reach, so
// that cells far apart still show walls.
const int NeighborhoodSize = 3;
const float GroupAffinity = 2.0;

struct Neighborhood
{
    uint spheres[NeighborhoodSize];
    float ranks[NeighborhoodSize];
};

Neighborhood createNeighborhood()
{
    Neighborhood result;
    for (int i = 0; i < NeighborhoodSize; ++i) {
        result.spheres[i] = NoEntry;
        result.ranks[i] = 1e20;
    }
    return result;
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
    tissue.brightness += sphere.details.x * weight;
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
    addMaterial(tissue, sphere, weight);
}

// A cell that ranks before an entry takes its place, the displaced entry moves on to the following places. A cell found again with a
// stronger group moves up.
void addToNeighborhood(inout Neighborhood neighborhood, uint index, Sphere sphere, vec2 position, float groupDominance)
{
    vec2 offset = position - sphere.centerRadius.xy;
    float rank = dot(offset, offset) - GroupAffinity * groupDominance;
    for (int i = 0; i < NeighborhoodSize; ++i) {
        if (neighborhood.spheres[i] == index) {
            if (rank >= neighborhood.ranks[i]) {
                return;
            }
            for (int j = i; j + 1 < NeighborhoodSize; ++j) {
                neighborhood.spheres[j] = neighborhood.spheres[j + 1];
                neighborhood.ranks[j] = neighborhood.ranks[j + 1];
            }
            neighborhood.spheres[NeighborhoodSize - 1] = NoEntry;
            neighborhood.ranks[NeighborhoodSize - 1] = 1e20;
            break;
        }
    }
    for (int i = 0; i < NeighborhoodSize; ++i) {
        if (rank < neighborhood.ranks[i]) {
            uint displacedSphere = neighborhood.spheres[i];
            float displacedRank = neighborhood.ranks[i];
            neighborhood.spheres[i] = index;
            neighborhood.ranks[i] = rank;
            index = displacedSphere;
            rank = displacedRank;
        }
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
// Each cell is a neuron with a small body in its middle and a dendrite towards each connected cell. A dendrite reaches the wall between the
// cells and branches there, so that it interweaves with the dendrite of the neighbor. Body and dendrites become brighter with the activity
// of the neuron, see ObjectVertexData::activity in Engine/Interface/GeometryBuffers.h, and the body shines with a soft halo. Sizes are given
// in world units.
const float SomaRadius = 0.065;
const float DendriteHalfWidth = 0.014;
const float DendriteReach = 0.46;  // Share of the distance between the cells

// Glow at rest and at full activity. The light of the body grows faster than linear with the activity, so that weak activity stays dim.
const float SomaGlow = 0.06;
const float SomaActivityGlow = 0.9;
const float SomaHaloGlow = 0.3;
const float DendriteGlow = 0.04;
const float DendriteActivityGlow = 0.18;

// Body and dendrites displace the cytoplasm and absorb light themselves
const float NeuronClearing = 0.3;
const float NeuronDensity = 0.3;

// The dendrites at a position are kept for each of their cells. Only those of the cells in the neighborhood are shown, so that a cell hidden
// below a body hides its dendrites as well, see PolygonDominance.
const int MaxDendriteCells = 4;

struct Dendrites
{
    uint cells[MaxDendriteCells];
    float coverages[MaxDendriteCells];
    vec3 lights[MaxDendriteCells];
};

Dendrites createDendrites()
{
    Dendrites result;
    for (int i = 0; i < MaxDendriteCells; ++i) {
        result.cells[i] = NoEntry;
        result.coverages[i] = 0.0;
        result.lights[i] = vec3(0.0);
    }
    return result;
}

// The dendrites of a cell overlap at its body, where their light does not add up. A further cell takes the place of the weakest one.
void addDendriteOfCell(inout Dendrites dendrites, uint cell, float coverage, vec3 light)
{
    int weakest = 0;
    for (int i = 0; i < MaxDendriteCells; ++i) {
        if (dendrites.cells[i] == cell) {
            dendrites.coverages[i] = max(dendrites.coverages[i], coverage);
            dendrites.lights[i] = max(dendrites.lights[i], light);
            return;
        }
        if (dendrites.coverages[i] < dendrites.coverages[weakest]) {
            weakest = i;
        }
    }
    if (coverage > dendrites.coverages[weakest]) {
        dendrites.cells[weakest] = cell;
        dendrites.coverages[weakest] = coverage;
        dendrites.lights[weakest] = light;
    }
}

void getVisibleDendrites(Dendrites dendrites, Neighborhood neighborhood, out float coverage, out vec3 light)
{
    coverage = 0.0;
    light = vec3(0.0);
    for (int i = 0; i < MaxDendriteCells; ++i) {
        for (int j = 0; j < NeighborhoodSize; ++j) {
            if (dendrites.cells[i] != NoEntry && dendrites.cells[i] == neighborhood.spheres[j]) {
                coverage = max(coverage, dendrites.coverages[i]);
                light = max(light, dendrites.lights[i]);
            }
        }
    }
}

// Lines thinner than a pixel keep their brightness spread over a pixel
float getLineCoverage(float distance, float halfWidth)
{
    float pixelSize = getPixelSize();
    float spreadHalfWidth = max(halfWidth, 0.55 * pixelSize);
    float edge = max(0.8 * pixelSize, 0.003);
    return clamp(0.5 - (distance - spreadHalfWidth) / edge, 0.0, 1.0) * halfWidth / spreadHalfWidth;
}

bool isNeuron(Sphere sphere)
{
    return sphere.neuron.x > 0.5;
}

// The light of a neuron is the hue of its cell, slightly whitened
vec3 getNeuronColor(Sphere cell)
{
    return mix(getSaturatedHue(cell.hueMaterial.rgb), vec3(1.0), 0.3);
}

void addBranch(inout float coverage, vec2 position, vec2 start, vec2 end, float startHalfWidth, float endHalfWidth)
{
    vec2 segment = end - start;
    float parameter = clamp(dot(position - start, segment) / dot(segment, segment), 0.0, 1.0);
    coverage = max(coverage, getLineCoverage(length(position - start - segment * parameter), mix(startHalfWidth, endHalfWidth, parameter)));
}

// The dendrite bends once, forks at its tip and sends a twig to the side. Its shape derives from the seeds of both cells and therefore
// stays the same over time, its branches taper towards their tips.
void addDendrite(inout Dendrites dendrites, uint cellIndex, Sphere cell, Sphere neighbor, vec2 position)
{
    vec2 axis = neighbor.centerRadius.xy - cell.centerRadius.xy;
    float distance = length(axis);
    if (distance < 1e-4) {
        return;
    }
    vec2 direction = axis / distance;
    vec2 offset = position - cell.centerRadius.xy;
    vec2 local = vec2(dot(offset, direction), dot(offset, vec2(-direction.y, direction.x)));
    float reach = DendriteReach * distance;
    if (local.x < -SomaRadius || local.x > reach + 0.1 || abs(local.y) > 0.25) {
        return;
    }
    float seed = hash(vec2(cell.details.y * 13.0, neighbor.details.y * 7.0));
    float side = seed < 0.5 ? 1.0 : -1.0;
    vec2 start = vec2(0.6 * SomaRadius, 0.0);
    vec2 bend = vec2(0.5 * reach, 0.03 * side);
    vec2 tip = vec2(reach, -0.02 * side);
    vec2 twig = bend + vec2(0.11, 0.12 * side * (0.8 + 0.4 * seed));
    float coverage = 0.0;
    addBranch(coverage, local, start, bend, DendriteHalfWidth, 0.01);
    addBranch(coverage, local, bend, tip, 0.01, 0.006);
    addBranch(coverage, local, bend, twig, 0.007, 0.0035);
    addBranch(coverage, local, twig, twig + vec2(0.06, 0.03 * side), 0.0035, 0.002);
    addBranch(coverage, local, tip, tip + vec2(0.07, 0.05), 0.005, 0.0025);
    addBranch(coverage, local, tip, tip + vec2(0.07, -0.05), 0.005, 0.0025);
    coverage *= frame.tissue.y;
    vec3 light = getNeuronColor(cell) * coverage * (DendriteGlow + DendriteActivityGlow * cell.neuron.y);

    if (coverage > 0.0) {
        addDendriteOfCell(dendrites, cellIndex, coverage, light);
    }
}
)"
                                                R"(
// The bounding boxes cover the range of each group, hence the ray finds all groups that reach the pixel. Each group reaches zero density
// at the border of its bounding box, so that the tissue stays continuous.
Tissue gatherTissue(vec2 position, out Neighborhood neighborhood, out Dendrites dendrites)
{
    TissueBlend blend = createTissueBlend();
    neighborhood = createNeighborhood();
    dendrites = createDendrites();
    float totalDensity = 0.0;
    vec2 totalGradient = vec2(0.0);
    rayQueryEXT query;
    rayQueryInitializeEXT(query, scene, gl_RayFlagsNoneEXT, 0xffu, vec3(position, CameraHeight), 0.0, vec3(0.0, 0.0, -1.0), 2.0 * CameraHeight);
    while (rayQueryProceedEXT(query)) {
        uint kind = uint(rayQueryGetIntersectionGeometryIndexEXT(query, false)) + 1u;
        uint primitive = uint(rayQueryGetIntersectionPrimitiveIndexEXT(query, false));

        // Each group starts with its own contribution, which enters the total, the cells of connections and triangles follow
        Tissue group = createTissue();
        float polygonBonus = 0.0;
        if (kind == Kind_Sphere) {
            Sphere sphere = spheres[primitive];
            addCell(group, sphere, position, 1.0);
            totalDensity += group.density;
            totalGradient += group.gradient;
            vec2 offset = position - sphere.centerRadius.xy;
            if (dot(offset, offset) < square(getMaxCellKernelRadius(sphere))) {
                addToNeighborhood(neighborhood, primitive, sphere, position, group.density);
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
            bool onStrand = group.density > 0.0;
            addCell(group, a, position, 1.0 - getSolidSmoothing(a));
            addCell(group, b, position, 1.0 - getSolidSmoothing(b));
            if (onStrand) {
                addToNeighborhood(neighborhood, indexA, a, position, group.density);
                addToNeighborhood(neighborhood, indexB, b, position, group.density);
            }
            if (isNeuron(a) && isNeuron(b) && frame.tissue.y > 0.0) {
                addDendrite(dendrites, indexA, a, b, position);
                addDendrite(dendrites, indexB, b, a, position);
            }
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
            bool onFill = group.density > 0.0;
            bool isTissue = max(a.hueMaterial.w, max(b.hueMaterial.w, c.hueMaterial.w)) == Material_Tissue;
            polygonBonus = isTissue ? PolygonDominance * group.density / FillWeight : 0.0;
            addCell(group, a, position, 1.0 - getSolidSmoothing(a));
            addCell(group, b, position, 1.0 - getSolidSmoothing(b));
            addCell(group, c, position, 1.0 - getSolidSmoothing(c));
            if (onFill) {
                addToNeighborhood(neighborhood, indexA, a, position, group.density + polygonBonus);
                addToNeighborhood(neighborhood, indexB, b, position, group.density + polygonBonus);
                addToNeighborhood(neighborhood, indexC, c, position, group.density + polygonBonus);
            }
        }
        addGroup(blend, group, group.density + polygonBonus);
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
    float grain;
    float organelle;
    vec3 cellColor;
    float cellColorWeight;
    float soma;
    vec3 somaLight;
};

// The distance is measured from the side of the nearest cell, see Neighborhood for the ranks. A wall only appears where it lies in the
// middle between both cells as well. The ranks of a cell lying on another body change quickly at the edge of its group, where the wall
// would otherwise enclose the cell like a bubble. The waves stay with the cells, since they derive from the seeds of both.
float getWallDistance(Sphere cell, Sphere neighbor, float cellRank, float neighborRank, vec2 position)
{
    vec2 axis = neighbor.centerRadius.xy - cell.centerRadius.xy;
    float centerDistance = max(length(axis), 1e-4);
    vec2 middle = (cell.centerRadius.xy + neighbor.centerRadius.xy) * 0.5;
    float wallDistance = (neighborRank - cellRank) / (2.0 * centerDistance);
    float middleDistance = dot(middle - position, axis / centerDistance);

    // Both cells have to see the same wall, hence the waves are defined in an orientation that does not depend on which one is nearer
    float orientation = cell.details.y < neighbor.details.y ? 1.0 : -1.0;
    vec2 direction = axis / centerDistance * orientation;
    float along = dot(position - middle, vec2(-direction.y, direction.x));
    float pairSeed = cell.details.y + neighbor.details.y;
    float wave = (valueNoise(vec2(along * WallWaveFrequency + pairSeed * 53.0, pairSeed * 31.0)) - 0.5) * WallWaveAmplitude;
    return max(abs(wallDistance * orientation - wave), abs(middleDistance * orientation - wave));
}

// The arrangement of the organelles derives from the seed of the cell and therefore stays with the cell
void addOrganelles(inout CellDetails details, Sphere cell, vec2 local, float visibility)
{
    float radius = cell.centerRadius.w;
    float seed = cell.details.y;
    float pixelSize = getPixelSize();
    for (int i = 0; i < OrganellesPerCell; ++i) {
        float angle = TwoPi * (seed + (float(i) + 0.6 * hash(vec2(seed, float(i) + 0.5))) / float(OrganellesPerCell));
        float distanceToCenter = radius * MaxOrganelleDistance * sqrt(hash(vec2(seed, float(i) + 1.5)));
        float size = radius * mix(MinOrganelleSize, MaxOrganelleSize, hash(vec2(seed, float(i) + 2.5)));
        float distance = length(local - distanceToCenter * vec2(cos(angle), sin(angle)));

        // Organelles of few pixels would only appear as speckles
        float organelleVisibility = smoothstep(1.5, 3.5, size / pixelSize) * visibility;
        float edge = max(OrganelleEdge * size, pixelSize);
        details.organelle = max(details.organelle, (1.0 - smoothstep(size - edge, size, distance)) * organelleVisibility);
    }
}

// The body of the neuron lies in the middle of the cell, see addDendrite
void addSoma(inout CellDetails details, Sphere cell, vec2 local, float visibility)
{
    float pixelSize = getPixelSize();
    float distance = length(local);
    float somaVisibility = smoothstep(0.8, 2.0, SomaRadius / pixelSize) * visibility;
    details.soma = clamp(0.5 - (distance - SomaRadius) / max(0.8 * pixelSize, 0.004), 0.0, 1.0) * somaVisibility;
    float intensity = pow(cell.neuron.y, 1.5);
    float halo = exp(-square(distance / (2.0 * SomaRadius))) * somaVisibility;
    details.somaLight = getNeuronColor(cell) * (details.soma * (SomaGlow + SomaActivityGlow * intensity) + SomaHaloGlow * intensity * halo);
}

// The cell at a position is the nearest one, the walls lie where the nearest and the second nearest cell are equally far away. The walls of
// the cells appear with the tissue, those of the solids and the inner details only when they span several pixels.
CellDetails getCellDetails(Neighborhood neighborhood, vec2 position)
{
    CellDetails result = CellDetails(0.0, 0.0, 0.0, vec3(0.0), 0.0, 0.0, vec3(0.0));
    if (neighborhood.spheres[0] == NoEntry) {
        return result;
    }
    Sphere cell = spheres[neighborhood.spheres[0]];
    bool solid = cell.hueMaterial.w == Material_Solid;
    float wallWidth = max(0.035, 0.7 * getPixelSize());
    float wallVisibility = solid ? getSolidDetailVisibility() : frame.tissue.w;
    for (int i = 1; i < NeighborhoodSize; ++i) {
        if (neighborhood.spheres[i] != NoEntry) {
            Sphere neighbor = spheres[neighborhood.spheres[i]];
            float wallDistance = getWallDistance(cell, neighbor, neighborhood.ranks[0], neighborhood.ranks[i], position);
            result.wall = max(result.wall, exp(-square(wallDistance / wallWidth)) * wallVisibility);
        }
    }
    if (solid) {
        return result;
    }
    float visibility = frame.tissue.y;
    result.cellColor = getSaturatedHue(cell.hueMaterial.rgb);
    result.cellColorWeight = CellColorSharpness * (1.0 - visibility);
    if (visibility <= 0.0) {
        return result;
    }
    vec2 local = position - cell.centerRadius.xy;
    result.grain = (valueNoise(local / cell.centerRadius.w * CytoplasmGrain + cell.details.y * 29.0) - 0.5) * visibility;
    addOrganelles(result, cell, local, visibility);
    if (isNeuron(cell)) {
        addSoma(result, cell, local, visibility);
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
vec3 shadeTissue(Tissue tissue, Neighborhood neighborhood, Dendrites dendrites, vec2 position, vec3 medium, vec3 sourceLight)
{
    float closeUpWeight = getCloseUpWeight();
    float excess = max(tissue.density - DensityThreshold, 0.0);
    float edgeDistance = excess / max(length(tissue.gradient), 1e-4);
    float outline = exp(-square(edgeDistance / max(0.05, 0.8 * getPixelSize()))) * closeUpWeight;

    float totalExcess = max(tissue.totalDensity - DensityThreshold, 0.0);
    float thicknessGain = mix(ThicknessGain, SmoothSolidThicknessGain, tissue.solid * (1.0 - getSolidDetailVisibility()));
    float thickness = mix(1.0, 1.0 - exp(-totalExcess * thicknessGain), closeUpWeight);
    vec2 thicknessGradient = closeUpWeight * thicknessGain * exp(-totalExcess * thicknessGain) * tissue.totalGradient;

    CellDetails details = getCellDetails(neighborhood, position);
    vec3 color = mix(getTissueColor(tissue), details.cellColor, details.cellColorWeight);
    float dendriteCoverage;
    vec3 dendriteLight;
    getVisibleDendrites(dendrites, neighborhood, dendriteCoverage, dendriteLight);
    float neuron = max(details.soma, dendriteCoverage);
    float cytoplasm = thickness * (1.0 + 0.35 * details.grain) * (1.0 - OrganelleShade * details.organelle) * (1.0 - NeuronClearing * neuron);

    float opticalDensity = cytoplasm + outline * 0.8 + details.wall * 0.6 + details.organelle * OrganelleDensity + NeuronDensity * neuron;
    vec3 transmittance = exp(-Absorption * opticalDensity * getAbsorbedShare(color));
    vec3 fluorescence = color * tissue.brightness * (CytoplasmGlow * cytoplasm + OutlineGlow * outline + WallGlow * details.wall);
    vec3 glow = max(details.somaLight, dendriteLight) + sourceLight * TissueSourceLight * thickness * color;
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

    // Zoomed out, the rasterized objects replace the tissue, whose scene is not built then
    float tissueVisibility = frame.tissue.w;
    vec3 background = getBackgroundRadiance(position);
    vec3 sourceLight = isSourceLightActive() ? getSourceLight(position) : vec3(0.0);
    vec3 medium = shadeGas(position, background, sourceLight) + getEnergyParticleRadiance(position);
    vec3 color = medium;
#ifndef WITHOUT_TISSUE
    if (tissueVisibility > 0.0) {
        Neighborhood neighborhood;
        Dendrites dendrites;
        Tissue tissue = gatherTissue(position, neighborhood, dendrites);
        if (tissue.weight > 0.0) {
            float coverage = getCoverage(tissue);
            color = mix(medium + getHalo(tissue), shadeTissue(tissue, neighborhood, dendrites, position, medium, sourceLight), coverage);
        }
        color = mix(medium, color, tissueVisibility);
    }
#endif

    // The rasterized objects scatter the light of the radiation sources as well, its strength is passed on to them in the alpha channel,
    // see Shaders/RayTracingMergeFS.h
    imageStore(sceneColor, pixel, vec4(getForeground(color, background), luminance(sourceLight) * TissueSourceLight));
}
)";
}
