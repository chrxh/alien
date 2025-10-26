#pragma once

#include <string_view>

namespace Shaders
{
    std::string_view const LineFS = R"(
#version 330 core
in vec3 fragColor;
out vec4 FragColor;

void main()
{
    // Use the lit color from geometry shader
    FragColor = vec4(fragColor, 1.0);
}
)";
}
