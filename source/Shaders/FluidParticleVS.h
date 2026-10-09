#pragma once

#include <string_view>

namespace Shaders
{
    std::string_view const FluidParticleVS = R"(
#version 450
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;
layout (location = 2) in float aGlow;

out vec3 vColor;

uniform vec2 worldSize;
uniform vec2 rectUpperLeft;
uniform float zoom;
uniform float radius;
uniform vec2 viewportSize;
uniform bool onBackground;
uniform float renderScale;

const float CoreHiddenZoom = 2.0;
const float CoreDetailZoom = 7.0;

void main()
{
    // Transform world position to normalized device coordinates
    vec2 relativePos = aPos.xy - rectUpperLeft;
    vec2 screenPos = relativePos * zoom;
    vec2 ndc = (screenPos / viewportSize) * 2.0 - 1.0;
    ndc.y = -ndc.y; // Flip Y coordinate
    gl_Position = vec4(ndc, 0.0, 1.0);

    float ballSize;
    float coreVisibility = 1.0;
    if (onBackground) {
        ballSize = 15.0;
    } else {
        float screenZoom = zoom / renderScale;
        ballSize = screenZoom < CoreDetailZoom ? 0.0 : 0.2;
        coreVisibility = clamp((screenZoom - CoreHiddenZoom) / (CoreDetailZoom - CoreHiddenZoom), 0.0, 1.0);
    }

    float g = clamp(aGlow, 0.0, 1.0);
    if (onBackground) {
        vColor = aColor * mix(1.0, 4.0, g);
    } else {
        ballSize = mix(ballSize, 1.0, g);
        vColor = aColor * mix(coreVisibility, 20.0, g);
    }

    gl_PointSize = radius * ballSize;
}
)";
}
