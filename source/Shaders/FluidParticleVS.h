#pragma once

#include <string_view>

namespace Shaders
{
    std::string_view const FluidParticleVS = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;
layout (location = 2) in float aEnergy;

out vec3 vColor;

uniform vec2 worldSize;
uniform vec2 rectUpperLeft;
uniform float zoom;
uniform float radius;
uniform vec2 viewportSize;
uniform bool onBackground;

void main()
{
    vColor = aColor;
    
    // Transform world position to normalized device coordinates
    vec2 relativePos = aPos.xy - rectUpperLeft;
    vec2 screenPos = relativePos * zoom;
    vec2 ndc = (screenPos / viewportSize) * 2.0 - 1.0;
    ndc.y = -ndc.y; // Flip Y coordinate
    gl_Position = vec4(ndc, 0.0, 1.0);

    float ballSize;
    if (onBackground) {
        ballSize = 10.0;
    } else {
        if (zoom < 7.0) {
            ballSize = (aEnergy > 300.0) ? 0.2 : 0.0;
        } else {
            ballSize = 0.2;
        }
    }
    gl_PointSize = radius * ballSize;
}
)";
}
