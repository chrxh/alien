#pragma once

#include <string_view>

namespace Shaders
{
    // Copies the ray traced image into the render target, whose rows run from the bottom to the top
    std::string_view const RayTracingPresentFS = R"(
#version 450
in vec2 texCoord;
out vec4 FragColor;

uniform sampler2D inputTexture1;

void main()
{
    FragColor = vec4(texture(inputTexture1, vec2(texCoord.x, 1.0 - texCoord.y)).rgb, 1.0);
}
)";
}
