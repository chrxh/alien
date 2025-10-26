#pragma once

#include <string_view>

namespace Shaders
{
    std::string_view const SelectedObjectFS = R"(#version 330 core
out vec4 FragColor;

in vec2 gWorldPos;
in vec2 gQuadCoord;

uniform float zoom;
uniform vec2 worldSize;
uniform float radius;

void main()
{
    // Calculate distance from center using quad coordinates
    float dist = length(gQuadCoord);
    
    // Circle radius in normalized quad space (0.5 = edge)
    float outerRadius = 0.5;
    float innerRadius = outerRadius - 3.0 / zoom; // Thin circle (2 pixels)
    
    // Discard pixels outside the outer radius
    if (dist > outerRadius) {
        discard;
    }
    
    // Discard pixels inside the inner radius (creating a hollow circle)
    if (dist < innerRadius) {
        discard;
    }
    
    // Anti-aliasing for smooth edges
    float outerEdge = smoothstep(outerRadius - 1.5 / zoom, outerRadius, dist);
    float innerEdge = smoothstep(innerRadius, innerRadius + 1.5 / zoom, dist);
    float alpha = (1.0 - outerEdge) * innerEdge;
    
    // White circle
    FragColor = vec4(1.0, 1.0, 1.0, alpha);
}
)";
}
