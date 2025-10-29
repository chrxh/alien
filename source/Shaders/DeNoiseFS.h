#pragma once

#include <string_view>

namespace Shaders
{
    std::string_view const DeNoiseFS = R"(
#version 330 core
out vec4 FragColor;

in vec2 texCoord;

uniform sampler2D inputTexture1;
uniform vec2 viewportSize;

void main()
{
    vec2 texelSize = 1.0 / viewportSize;
    vec4 result = vec4(0.0);

    result += texture(inputTexture1, texCoord);
    result = max(result, texture(inputTexture1, texCoord + vec2(texelSize.x, 0)));
    result = max(result, texture(inputTexture1, texCoord + vec2(0, texelSize.y)));
    result = max(result, texture(inputTexture1, texCoord + vec2(texelSize.x, texelSize.y)));
    FragColor = result;
}
)";
}
