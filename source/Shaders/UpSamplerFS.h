#pragma once

#include <string_view>

namespace Shaders
{
    std::string_view const UpSamplerFS = R"(
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
    vec3 sum = vec3(0.0);
    float kernel[9] = float[9](1,1,1, 1,1,1, 1,1,1);
    float wsum = 9.0;

    float offset = (1.0 - 1.0 / scale) / 2;
    vec2 sourceTexCoord = texCoord / scale + vec2(offset, offset);
    FragColor = texture(inputTexture1, sourceTexCoord);


    //float targetTexSize = scale / 2;
    //vec2 targetTexCoord = (texCoord - vec2(0.5, 0.5)) / vec2(targetTexSize);
    //targetTexCoord = (targetTexCoord + vec2(1.0)) / 2;  // Normalize to [0, 1] range


    //vec2 texelSize = 1.0 / viewportSize;
    //int k = 0;
    //for (int y = -1; y <= 1; ++y) {
    //    for (int x = -1; x <= 1; ++x) {
    //        vec2 offset = vec2(x, y) * texelSize;
    //        sum += texture(inputTexture1, texCoord + offset).rgb * kernel[k++];
    //    }
    //}

    //FragColor = vec4(sum / wsum, 1.0);
}
)";
}
