#pragma once

#include <string_view>

namespace Shaders
{
    // Takes the maximum of the weighted first and the second texture of a pair. With four textures, the maxima of both pairs add up.
    std::string_view const MergeMaxFS = R"(
#version 450
out vec4 FragColor;

in vec2 texCoord;

uniform sampler2D inputTexture1;
uniform sampler2D inputTexture2;
uniform sampler2D inputTexture3;
uniform sampler2D inputTexture4;
uniform vec2 viewportSize;
uniform float zoom;
uniform int mode;
uniform float colorFactor1;
uniform int numTextures;

void main()
{
    vec3 finalColor = max(texture(inputTexture1, texCoord).rgb * colorFactor1, texture(inputTexture2, texCoord).rgb);
    if (numTextures >= 4) {
        finalColor += max(texture(inputTexture3, texCoord).rgb * colorFactor1, texture(inputTexture4, texCoord).rgb);
    }
    FragColor = vec4(finalColor, 1.0);
}
)";
}
