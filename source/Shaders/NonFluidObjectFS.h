#pragma once

#include <string_view>

namespace Shaders
{
    std::string_view const NonFluidObjectFS = R"(
#version 330 core
out vec4 FragColor;

in vec3 vColor;

#define PI 3.1415926538

void main()
{
    // Calculate distance from center of point
    vec2 coord = gl_PointCoord - vec2(0.5, 0.5);
    float dist = length(coord);
    
    // Discard pixels outside the circle
     if (dist > 0.5) {
         discard;
     }
    
    // Foreground objects - smooth circles with angle-based factor
    float angle = atan(coord.y, coord.x) * 180.0 / PI;
    angle += 45.0;
    if (angle > 180.0) {
        angle -= 360.0;
    }
    float factor = min(1.0, 60.0 / (abs(angle) + 1.0));
    float alpha = max(0.0, (1.0 - smoothstep(0.3, 0.5, dist)) * smoothstep(0.0, 0.1, dist) * factor);
    FragColor = vec4(vColor * alpha, 1.0);
}
)";
}
