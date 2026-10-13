#pragma once

#include <string_view>

namespace Shaders
{
    std::string_view const RayTracingPresentVS = R"(
#version 450
out vec2 texCoord;

void main()
{
    vec2 position = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    texCoord = position;
    gl_Position = vec4(position * 2.0 - 1.0, 0.0, 1.0);
}
)";
}
