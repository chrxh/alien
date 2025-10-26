#pragma once

#include <string_view>

namespace Shaders
{
    constexpr std::string_view SelectedObjectVS = R"(#version 330 core
layout (location = 0) in vec2 aPos;

out vec2 vWorldPos;

uniform vec2 worldSize;
uniform vec2 rectUpperLeft;
uniform float zoom;
uniform float radius;
uniform vec2 viewportSize;

void main()
{
    vWorldPos = aPos;
    
    // Transform world position to normalized device coordinates
    vec2 relativePos = aPos - rectUpperLeft;
    vec2 screenPos = relativePos * zoom;
    vec2 ndc = (screenPos / viewportSize) * 2.0 - 1.0;
    ndc.y = -ndc.y; // Flip Y coordinate
    gl_Position = vec4(ndc, 0.0, 1.0);
}
)";
}
