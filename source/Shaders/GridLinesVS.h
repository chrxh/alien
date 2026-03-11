#pragma once

#include <string_view>

namespace Shaders
{
    std::string_view const GridLinesVS = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoord;

out vec2 texCoord;

void main()
{
    vec3 flipped = vec3(aPos.x, -aPos.y, 0.0);   // Flip Y coordinate
    gl_Position = vec4(flipped, 1.0);
    texCoord = aTexCoord;
}
)";
}
