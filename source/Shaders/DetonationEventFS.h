#pragma once

#include <string_view>

namespace Shaders
{
    std::string_view const DetonationEventFS = R"(
#version 330 core
in vec2 quadCoord;
in float fragRadius;
out vec4 FragColor;

uniform float zoom;

void main()
{
    // Calculate distance from center of the circle
    float dist = length(quadCoord);
    
    // Discard fragments outside the circle
    if (dist > 1.0) {
        discard;
    }
    
    // Yellow color for detonation
    vec3 color = vec3(1.0, 1.0, 0.0);
    
    // Add alpha falloff from center to edge for a glowing effect
    float alpha = 1.0 - smoothstep(0.0, 1.0, dist);
    alpha = alpha * 0.8 + 0.2; // Ensure minimum visibility
    
    FragColor = vec4(color, alpha);
}
)";
}
