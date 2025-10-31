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
uniform float scale;

void main()
{
    vec3 color = vec3(0.0);
    float kernel[9] = float[9](1, 1, 1,
                               1, 1, 1,
                               1, 1, 1);
    float weightSum = 9.0; // sum(kernel)
    vec2 texelSize = 1.0 / viewportSize;

    float targetTexSize = scale / 2;
    vec2 targetTexCoord = (texCoord - vec2(0.5, 0.5)) / vec2(targetTexSize);
    targetTexCoord = (targetTexCoord + vec2(1.0)) / 2;  // Normalize to [0, 1] range

    if (targetTexCoord.x >= 0.0f && targetTexCoord.x <= 1.0f &&
        targetTexCoord.y >= 0.0f && targetTexCoord.y <= 1.0f) {

        int i = 0;
        for (int y = -1; y <= 1; ++y) {
            for (int x = -1; x <= 1; ++x) {
                vec2 offset = vec2(x, y) * texelSize;
                color += texture(inputTexture1, targetTexCoord + offset).rgb * kernel[i++];
            }
        }
        FragColor = vec4(color / weightSum, 1.0);
    }
    else {
        FragColor = vec4(0.0, 0.0, 0.0, 1.0);
    }
}
)";
}
