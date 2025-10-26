#pragma once

#include <string_view>

namespace Shaders
{
    constexpr std::string_view UpSamplerFS = R"(#version 330 core
out vec4 FragColor;

in vec2 texCoord;

uniform sampler2D inputTexture1;
uniform sampler2D inputTexture2;
uniform sampler2D inputTexture3;
uniform vec2 viewportSize;
uniform float zoom;

void main()
{
    vec3 sum = vec3(0.0);
    float kernel[9] = float[9](1,2,1, 2,4,2, 1,2,1);
    float wsum = 16.0;

    vec2 texelSize = 1.0 / viewportSize;
    int k = 0;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            vec2 offset = vec2(x, y) * texelSize;
            sum += texture(inputTexture1, texCoord + offset).rgb * kernel[k++];
        }
    }

    FragColor = vec4(sum / wsum, 1.0);
}
)";
}
