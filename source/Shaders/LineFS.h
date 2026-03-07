#pragma once

#include <string_view>

namespace Shaders
{
    std::string_view const LineFS = R"(
#version 330 core
in vec3 fragColor;
in float edgeDist;
in float lineAlpha;
in float coreEdge;
out vec4 FragColor;

void main()
{
    // Core region at full brightness, fade only in the AA margin
    float edgeAlpha = 1.0 - smoothstep(coreEdge, 1.0, abs(edgeDist));
    float alpha = edgeAlpha * lineAlpha;
    FragColor = vec4(fragColor * alpha, 1.0);
}
)";
}
