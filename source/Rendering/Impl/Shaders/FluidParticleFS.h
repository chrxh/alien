#pragma once

#include <string_view>

namespace Shaders
{
    std::string_view const FluidParticleFS = R"(
#version 450
out vec4 FragColor;

in vec3 vColor;
in vec2 vAxis;
in float vElongation;
uniform float zoom;
uniform bool onBackground;

void main()
{
    // Distance from the center of the ellipse relative to the point size. The point grows with the square root of the elongation, so that the
    // ellipse becomes longer along the axis and narrower across it by this factor and keeps the area of the circle.
    vec2 coord = gl_PointCoord - vec2(0.5, 0.5);
    float dist = length(vec2(dot(coord, vAxis), dot(coord, vec2(-vAxis.y, vAxis.x)) * vElongation));

    // Discard pixels outside the ellipse
    if (dist > 0.5) {
        discard;
    }
    
    // Simple falloff for alpha based on distance from center
    float alpha;
    if (onBackground) {
        alpha = (1.0 - smoothstep(0.0, 0.5, dist)) * 0.15;
    } else {
        alpha = 1.0 - smoothstep(0.2, 0.5, dist);
    }
    
    FragColor = vec4(vColor * alpha, 1.0);
}
)";
}
