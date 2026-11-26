#pragma once

#include <string_view>

namespace Shaders
{
    std::string_view const TriangleGS = R"(
#version 450
layout (triangles) in;
layout (triangle_strip, max_vertices = 3) out;

in vec3 vertexColor[];
in float vertexSolid[];
out vec3 fragColor;

uniform float zoom;
uniform bool proportionalCellBrightness;
uniform float cellBrightness;
//uniform float lightAngle;

#define PI 3.1415926538

// The heights of the objects vary randomly, so the lighting of small triangles is random as well. Solids are lit evenly with about the mean of
// this lighting, so that their polygons and connections merge into a smooth area.
const float SolidLighting = 0.94;

// See Shaders/NonFluidObjectVS.h
float getMetaballBrightness(float brightness)
{
    return brightness * smoothstep(0.05, 0.55, brightness);
}

// The metaballs step darkens dim objects far more than bright ones, see Shaders/MetaballsFS.h. Where the rasterized objects replace the ray
// traced tissue, whose brightness is proportional to the energy of the cells, the color of a cell is chosen such that it leaves that step with
// the given share of its brightness. Dim cells are brightened for it instead of fading out.
vec3 getCellColor(vec3 color)
{
    float brightness = color.r + color.g + color.b;
    if (!proportionalCellBrightness || brightness <= 0.0) {
        return color;
    }
    float target = brightness * cellBrightness;
    float low = 0.0;
    float high = max(brightness, 1.0);
    for (int i = 0; i < 16; ++i) {
        float middle = 0.5 * (low + high);
        if (getMetaballBrightness(middle) < target) {
            low = middle;
        } else {
            high = middle;
        }
    }
    return color * high / brightness;
}

void main()
{
    // Create 3D positions for lighting calculation
    vec3 pos0 = vec3(gl_in[0].gl_Position.xy, gl_in[0].gl_Position.z * 10);
    vec3 pos1 = vec3(gl_in[1].gl_Position.xy, gl_in[1].gl_Position.z * 10);
    vec3 pos2 = vec3(gl_in[2].gl_Position.xy, gl_in[2].gl_Position.z * 10);
    
    // Calculate triangle normal using cross product
    vec3 edge1 = pos1 - pos0;
    vec3 edge2 = pos2 - pos0;
    vec3 normal = normalize(cross(edge1, edge2));
    
    // Light direction from front (camera looks at -Z, so light from +Z)
//    vec3 lightDir = normalize(vec3(sin(lightAngle / 180.0 * PI), cos(lightAngle / 180.0 * PI), -1.0));
    vec3 lightDir = normalize(vec3(1.0, 1.0, -1.0));
    
    // Calculate lighting (dot product of normal and light direction)
    // Clamp to [0, 1] range
    float lightIntensity = max(0.0, dot(normal, lightDir));
    //if (zoom < 3.0) {
    //    lightIntensity = 0.0;
    //}
    
    // Output the triangle with per-vertex lit colors (will be interpolated by GPU)
    for (int i = 0; i < 3; i++)
    {
        gl_Position = gl_in[i].gl_Position;
        // Keep the per triangle lighting subtle, the rounded shape of a body comes from the organic
        // surface filter. A strong term would show the triangulation as hard facets. The constant is
        // chosen so that the mean over both triangle windings stays at the one of the former term.
        vec3 color = vertexColor[i] * mix(0.9 + lightIntensity * 0.16, SolidLighting, vertexSolid[i]);
        fragColor = vertexSolid[i] > 0.5 ? color : getCellColor(color);
        EmitVertex();
    }
    EndPrimitive();
}
)";
}
