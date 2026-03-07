#pragma once

#include <string_view>

namespace Shaders
{
    std::string_view const LineFS = R"(
#version 330 core
in vec3 fragColor;
in float edgeDist;
in float lineAlpha;
out vec4 FragColor;

void main()
{
    // Smooth edge falloff for anti-aliasing
    float edgeAlpha = 1.0 - smoothstep(0.5, 1.0, abs(edgeDist));
    float alpha = edgeAlpha * lineAlpha;
    FragColor = vec4(fragColor * alpha, 1.0);
}
)";
}
