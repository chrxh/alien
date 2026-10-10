#pragma once

#include <string_view>

namespace Shaders
{
    std::string_view const TriangleFS = R"(
#version 450
in vec3 fragColor;
out vec4 FragColor;

void main()
{
    // Use the averaged color from geometry shader with some transparency
    FragColor = vec4(fragColor, 1.0);
}
)";
}
