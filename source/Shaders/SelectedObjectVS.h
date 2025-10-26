#pragma once

#include <string_view>

namespace Shaders
{
    std::string_view const SelectedObjectVS = R"(
#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in int aHasSignalRestriction;
layout (location = 2) in float aStartAngle;
layout (location = 3) in float aEndAngle;

out vec2 vWorldPos;
flat out int vHasSignalRestriction;
flat out float vStartAngle;
flat out float vEndAngle;

uniform vec2 worldSize;
uniform vec2 rectUpperLeft;
uniform float zoom;
uniform float radius;
uniform vec2 viewportSize;

void main()
{
    vWorldPos = aPos;
    vHasSignalRestriction = aHasSignalRestriction;
    vStartAngle = aStartAngle;
    vEndAngle = aEndAngle;
    
    // Transform world position to normalized device coordinates
    vec2 relativePos = aPos - rectUpperLeft;
    vec2 screenPos = relativePos * zoom;
    vec2 ndc = (screenPos / viewportSize) * 2.0 - 1.0;
    ndc.y = -ndc.y; // Flip Y coordinate
    gl_Position = vec4(ndc, 0.0, 1.0);
}
)";
}
