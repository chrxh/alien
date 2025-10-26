#pragma once

#include <string_view>

namespace Shaders
{
    std::string_view const DownSamplerFS = R"(
#version 330 core
out vec4 FragColor;

in vec2 texCoord;

uniform sampler2D inputTexture1;
uniform sampler2D inputTexture2;
uniform sampler2D inputTexture3;
uniform vec2 viewportSize;
uniform float zoom;

void main()
{
    vec3 color = vec3(0.0);
    float kernel[9] = float[9](1, 2, 1,
                               2, 4, 2,
                               1, 2, 1);
    float weightSum = 16.0; // sum(kernel)

    vec2 texelSize = 1.0 / viewportSize;
    int i = 0;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            vec2 offset = vec2(x, y) * texelSize;
            color += texture(inputTexture1, texCoord + offset).rgb * kernel[i++];
        }
    }

    FragColor = vec4(color / weightSum, 1.0);
}
)";
}
