#pragma once

#include <string_view>

namespace Shaders
{
    std::string_view const ObjectVS = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;
layout (location = 2) in int state;

out vec3 vColor;

uniform vec2 worldSize;
uniform vec2 rectUpperLeft;
uniform float zoom;
uniform float radius;
uniform vec2 viewportSize;

void main()
{
    // Transform world position to normalized device coordinates
    vec2 relativePos = aPos.xy - rectUpperLeft;
    vec2 screenPos = relativePos * zoom;
    vec2 ndc = (screenPos / viewportSize) * 2.0 - 1.0;
    ndc.y = -ndc.y; // Flip Y coordinate
    
    // Cells are rendered in front of lines (apply negative bias to bring forward)
    gl_Position = vec4(ndc, aPos.z, 1.0);

    bool isInTriangleOrQuad = ((state >> 16) & 0x1) == 1;
    if (isInTriangleOrQuad && zoom < 3.0) {
        // Discard cells that are part of triangles/quads when zoomed out to avoid Moire patterns
        gl_Position = vec4(-2.0, -2.0, -2.0, 1.0);
    }
    
    int signalState = (state >> 8) & 0xFF;
    if (signalState == 1 && zoom > 6.0f) {
        vColor = mix(aColor, vec3(1.0), 0.1);
        gl_PointSize = radius * 0.5;
    } else if (signalState == 2 && zoom > 6.0f) {
        vColor = mix(aColor, vec3(1.0), 0.2);
        gl_PointSize = radius * 0.6;
    } else {
        vColor = aColor;
        gl_PointSize = radius * 0.4;
    }
}
)";
}
