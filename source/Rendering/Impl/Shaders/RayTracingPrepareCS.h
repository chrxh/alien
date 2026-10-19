#pragma once

#include <string_view>

namespace Shaders
{
    // Derives the objects from the geometry buffers and creates the bounding boxes of the groups that form the tissue: each cell alone, each
    // connection together with its two cells and each triangle together with its corners
    std::string_view const RayTracingPrepareCS = R"(
Sphere createSphere(uint index)
{
    ObjectVertex object = objects[index];
    int objectType = getObjectType(object);
    vec3 color = vec3(object.color[0], object.color[1], object.color[2]);
    vec3 hue = getHue(color);

    float radius = CellRadius;
    float material = Material_Tissue;
    if (objectType == ObjectType_Solid) {
        radius = SolidRadius;
        material = Material_Solid;
    } else if (objectType == ObjectType_FreeCell) {
        radius = FreeCellRadius;
    } else if (objectType == ObjectType_Fluid) {
        material = Material_Inactive;
    }

    // The height of an object derives from its id and the id of its creature and therefore stays the same over time
    float seed = fract(object.pos[2] * 997.13 + 0.31);

    Sphere result;
    result.centerRadius = vec4(object.pos[0], object.pos[1], 0.0, radius);
    result.hueMaterial = vec4(hue, material);
    result.details = vec4(getBrightness(color), seed, float((object.state >> 16) & 1), 0.0);
    result.neuron = vec4(objectType == ObjectType_Cell ? 1.0 : 0.0, object.activity, 0.0, 0.0);
    return result;
}

// Without visible tissue, the scene only serves the shadows of the light of the radiation sources, which need the sizes of the objects close
// up
bool isSceneForShadowsOnly()
{
    return frame.tissue.w <= 0.0;
}

// The rays to a radiation source start at most about one node of the irradiance grid beyond its range around the area of the source
bool isInSourceLightRange(vec2 lower, vec2 upper)
{
    vec2 center = 0.5 * (lower + upper);
    float radius = 0.5 * length(upper - lower) + frame.gasLight.z;
    for (int i = 0; i < frame.numSourceLights.x; ++i) {
        SourceLight light = frame.sourceLights[i];
        light.centerRangeShape.xy = getNearestSourceCenter(light, center);
        if (length(getClosestSourcePoint(light, center) - center) <= light.centerRangeShape.z + radius) {
            return true;
        }
    }
    return false;
}

// A NaN as minimum deactivates the bounding box. The boxes extend over all heights, since the rays run vertically. Shadows only need the
// objects in the range of the sources.
Aabb createAabb(vec2 lower, vec2 upper, bool enabled)
{
    enabled = enabled && (!isSceneForShadowsOnly() || isInSourceLightRange(lower, upper));
    return Aabb(enabled ? lower.x : uintBitsToFloat(0x7fc00000u), lower.y, -1.0, upper.x, upper.y, 2.0);
}

float getCellExtent(Sphere sphere)
{
    float shadowExtent = sphere.centerRadius.w + getShadowBeamMargin();
    return isSceneForShadowsOnly() ? shadowExtent : max(getMaxCellKernelRadius(sphere), shadowExtent);
}

void main()
{
    uint index = gl_GlobalInvocationID.x;
    if (index < frame.counts.x) {
        Sphere sphere = createSphere(index);
        spheres[index] = sphere;
        vec2 center = sphere.centerRadius.xy;
        float extent = getCellExtent(sphere);
        bool enabled = sphere.hueMaterial.w != Material_Inactive;
        sphereAabbs[index] = createAabb(center - extent, center + extent, enabled);
    }
    if (index < frame.counts.y) {
        uint sphereA;
        uint sphereB;
        getCapsuleSpheres(index, sphereA, sphereB);
        Sphere a = createSphere(sphereA);
        Sphere b = createSphere(sphereB);
        float strandShadowExtent = StrandHalfWidth + getShadowBeamMargin();
        float strandExtent = isSceneForShadowsOnly() ? strandShadowExtent : max(getStrandKernelRadius(getStrandSolidSmoothing(a, b)), strandShadowExtent);
        float extent = max(strandExtent, max(getCellExtent(a), getCellExtent(b)));
        bool enabled = a.hueMaterial.w != Material_Inactive && b.hueMaterial.w != Material_Inactive;
        capsuleAabbs[index] =
            createAabb(min(a.centerRadius.xy, b.centerRadius.xy) - extent, max(a.centerRadius.xy, b.centerRadius.xy) + extent, enabled);
    }
    if (index < frame.counts.z) {
        uint sphereA;
        uint sphereB;
        uint sphereC;
        getTriangleSpheres(index, sphereA, sphereB, sphereC);
        Sphere a = createSphere(sphereA);
        Sphere b = createSphere(sphereB);
        Sphere c = createSphere(sphereC);
        float fillExtent = isSceneForShadowsOnly() ? 0.0 : getFillFalloff();
        float extent = max(fillExtent, max(getCellExtent(a), max(getCellExtent(b), getCellExtent(c))));
        bool enabled = a.hueMaterial.w != Material_Inactive && b.hueMaterial.w != Material_Inactive && c.hueMaterial.w != Material_Inactive;
        vec2 lower = min(min(a.centerRadius.xy, b.centerRadius.xy), c.centerRadius.xy);
        vec2 upper = max(max(a.centerRadius.xy, b.centerRadius.xy), c.centerRadius.xy);
        triangleAabbs[index] = createAabb(lower - extent, upper + extent, enabled);
    }
}
)";
}
