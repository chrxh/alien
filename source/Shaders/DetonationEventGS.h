#pragma once

#include <string_view>

namespace Shaders
{
    std::string_view const DetonationEventGS = R"(
#version 330 core
layout (points) in;
layout (triangle_strip, max_vertices = 4) out;

in float vertexRadius[];
in float vertexAge[];

// Relative to the detonation in world units, y points upwards on screen like gl_FragCoord
out vec2 localPos;
flat out vec2 centerPixelPos;
flat out float blastRadius;
flat out float age;

uniform vec2 viewportSize;
uniform float zoom;

// Reach of the light in multiples of the detonator radius
const float LightReach = 10.0;

void emitCorner(vec2 corner, vec4 center, float extent)
{
    vec2 ndcExtent = extent * zoom / viewportSize * 2.0;
    gl_Position = vec4(center.xy + corner * ndcExtent, 0.0, 1.0);
    localPos = corner * extent;
    centerPixelPos = (center.xy * 0.5 + 0.5) * viewportSize;
    blastRadius = vertexRadius[0];
    age = vertexAge[0];
    EmitVertex();
}

void main()
{
    vec4 center = gl_in[0].gl_Position;
    float extent = vertexRadius[0] * LightReach;

    emitCorner(vec2(-1.0, -1.0), center, extent);
    emitCorner(vec2(1.0, -1.0), center, extent);
    emitCorner(vec2(-1.0, 1.0), center, extent);
    emitCorner(vec2(1.0, 1.0), center, extent);
    EndPrimitive();
}
)";
}
