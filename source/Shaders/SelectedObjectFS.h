#pragma once

#include <string_view>

namespace Shaders
{
    std::string_view const SelectedObjectFS = R"(
#version 330 core
out vec4 FragColor;

in vec2 gWorldPos;
in vec2 gQuadCoord;

uniform float zoom;
uniform vec2 worldSize;
uniform float radius;

const float AlphaForCircle = 0.3;

void main()
{
    // Calculate distance from center using quad coordinates
    float dist = length(gQuadCoord);
    
    // Circle radius in normalized quad space (0.5 = edge)
    float outerRadius = 0.4;
    float innerRadius = outerRadius - 3.0 / zoom; // Thin circle (2 pixels)

    if (dist > outerRadius || dist < innerRadius) {
        discard;
    }

    FragColor = vec4(1.0, 1.0, 1.0, AlphaForCircle);
}
)";
}
