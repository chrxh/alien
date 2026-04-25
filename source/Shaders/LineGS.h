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
out vec2 lineCoord;
flat out float lineLength;
flat out float coreHalfWidth;
flat out float aaMargin;

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
    float segmentLength = length(p1.xy - p0.xy);
    vec2 dir = segmentLength > 0.0 ? (p1.xy - p0.xy) / segmentLength : vec2(1.0, 0.0);

    // Calculate perpendicular direction (for line thickness)
    vec2 perp = vec2(-dir.y, dir.x);

    // Core half-width in pixels (same as original line width)
    float coreWidth = max(zoom * 0.15 * 0.5, 0.5);

    // Add AA margin for smooth edges without shrinking the visible line
    float aaWidth = 1.5;

    float totalHalfWidth = coreWidth + aaWidth;
    vec2 offset = perp * totalHalfWidth;
    vec2 capOffset = dir * totalHalfWidth;

    // Calculate a perpendicular normal for lighting
    vec3 normal = normalize(vec3(-dir.y, dir.x, 0.0));

    // Light direction (same as triangles)
    vec3 lightDir = normalize(vec3(1.0, 1.0, -1.0));

    // Calculate lighting (dot product of normal and light direction)
    float lightIntensity = max(0.0, dot(normal, lightDir));

    // Apply lighting to colors (same formula as triangles)
    vec3 color0 = vertexColor[0] * (0.8 + lightIntensity * 0.2);
    vec3 color1 = vertexColor[1] * (0.8 + lightIntensity * 0.2);

    // Generate quad (4 vertices as triangle strip)
    // Vertex 0 (bottom-left)
    gl_Position = vec4(transform(p0.xy - capOffset - offset), p0.z, 1.0);
    fragColor = color0;
    lineCoord = vec2(-totalHalfWidth, -totalHalfWidth);
    lineLength = segmentLength;
    coreHalfWidth = coreWidth;
    aaMargin = aaWidth;
    EmitVertex();

    // Vertex 1 (top-left)
    gl_Position = vec4(transform(p0.xy - capOffset + offset), p0.z, 1.0);
    fragColor = color0;
    lineCoord = vec2(-totalHalfWidth, totalHalfWidth);
    lineLength = segmentLength;
    coreHalfWidth = coreWidth;
    aaMargin = aaWidth;
    EmitVertex();

    // Vertex 2 (bottom-right)
    gl_Position = vec4(transform(p1.xy + capOffset - offset), p1.z, 1.0);
    fragColor = color1;
    lineCoord = vec2(segmentLength + totalHalfWidth, -totalHalfWidth);
    lineLength = segmentLength;
    coreHalfWidth = coreWidth;
    aaMargin = aaWidth;
    EmitVertex();

    // Vertex 3 (top-right)
    gl_Position = vec4(transform(p1.xy + capOffset + offset), p1.z, 1.0);
    fragColor = color1;
    lineCoord = vec2(segmentLength + totalHalfWidth, totalHalfWidth);
    lineLength = segmentLength;
    coreHalfWidth = coreWidth;
    aaMargin = aaWidth;
    EmitVertex();

    EndPrimitive();
}
)";
}
