#pragma once

#include <string_view>

namespace Shaders
{
    std::string_view const DetonationEventVS = R"(
#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in float aRadius;
layout (location = 2) in float aAge;

out float vertexRadius;
out float vertexAge;

uniform vec2 rectUpperLeft;
uniform float zoom;
uniform vec2 viewportSize;

void main()
{
    vec2 relativePos = aPos - rectUpperLeft;
    vec2 screenPos = relativePos * zoom;
    vec2 ndc = (screenPos / viewportSize) * 2.0 - 1.0;
    ndc.y = -ndc.y; // Flip Y coordinate

    gl_Position = vec4(ndc, 0.0, 1.0);

    vertexRadius = aRadius;
    vertexAge = aAge;
}
)";
}
