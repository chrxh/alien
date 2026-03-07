#pragma once

#include <string_view>

namespace Shaders
{
    std::string_view const ObjectVS = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;
layout (location = 2) in int state;
layout (location = 3) in float signalChanges;

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

    bool isIsolated = ((state >> 16) & 0x1) == 1;
    if (!isIsolated && zoom < 10.0) {
        // Discard cells that have connections when zoomed out to avoid Moire patterns
        gl_Position = vec4(-2.0, -2.0, -2.0, 1.0);
    }
    
    vColor = mix(aColor, vec3(1.0), signalChanges * 0.2);
    gl_PointSize = radius * (0.4 + signalChanges * 0.2);
}
)";
}
