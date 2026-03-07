#pragma once

#include <string_view>

namespace Shaders
{
    std::string_view const LineGS = R"(
#version 330 core
layout (lines) in;
layout (triangle_strip, max_vertices = 4) out;

in vec3 vertexColor[];
out vec3 fragColor;
out float edgeDist;
out float lineAlpha;

uniform vec2 viewportSize;
uniform float zoom;

vec2 transform(vec2 v)
{
    v = v / viewportSize * 2.0 - 1.0;
    return vec2(v.x, -v.y);
}

void main()
{
    // Get the two vertices of the line in screen space
    vec4 p0 = gl_in[0].gl_Position;
    vec4 p1 = gl_in[1].gl_Position;
    
    // Calculate line direction in screen space
    vec2 dir = normalize(p1.xy - p0.xy);
    
    // Calculate perpendicular direction (for line thickness)
    vec2 perp = vec2(-dir.y, dir.x);
    
    // Line half-width in pixels with minimum to prevent sub-pixel disappearance
    float halfWidth = zoom * 0.15 * 0.5;
    float minHalfWidth = 1.0;
    float alpha = 1.0;
    if (halfWidth < minHalfWidth) {
        alpha = halfWidth / minHalfWidth;
        halfWidth = minHalfWidth;
    }
    vec2 offset = perp * halfWidth;
    
    // Create 3D positions for lighting calculation
    vec3 pos0 = vec3(p0.xyz);
    vec3 pos1 = vec3(p1.xyz);
    
    // Calculate line direction and a perpendicular normal for lighting
    vec3 lineDir = normalize(pos1 - pos0);
    vec3 normal = normalize(vec3(-lineDir.y, lineDir.x, 0.0));
    
    // Light direction (same as triangles)
    vec3 lightDir = normalize(vec3(1.0, 1.0, -1.0));
    
    // Calculate lighting (dot product of normal and light direction)
    float lightIntensity = max(0.0, dot(normal, lightDir));
    
    // Apply lighting to colors (same formula as triangles)
    vec3 color0 = vertexColor[0] * (0.8 + lightIntensity * 0.2);
    vec3 color1 = vertexColor[1] * (0.8 + lightIntensity * 0.2);
    
    // Generate quad (4 vertices as triangle strip)
    // Vertex 0 (bottom-left)
    gl_Position = vec4(transform(p0.xy - offset), p0.z, 1.0);
    fragColor = color0;
    edgeDist = -1.0;
    lineAlpha = alpha;
    EmitVertex();
    
    // Vertex 1 (top-left)
    gl_Position = vec4(transform(p0.xy + offset), p0.z, 1.0);
    fragColor = color0;
    edgeDist = 1.0;
    lineAlpha = alpha;
    EmitVertex();
    
    // Vertex 2 (bottom-right)
    gl_Position = vec4(transform(p1.xy - offset), p1.z, 1.0);
    fragColor = color1;
    edgeDist = -1.0;
    lineAlpha = alpha;
    EmitVertex();
    
    // Vertex 3 (top-right)
    gl_Position = vec4(transform(p1.xy + offset), p1.z, 1.0);
    fragColor = color1;
    edgeDist = 1.0;
    lineAlpha = alpha;
    EmitVertex();
    
    EndPrimitive();
}
)";
}
