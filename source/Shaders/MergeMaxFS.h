#pragma once

#include <string_view>

namespace Shaders
{
    std::string_view const MergeMaxFS = R"(
#version 330 core
out vec4 FragColor;

in vec2 texCoord;

uniform sampler2D inputTexture1;
uniform sampler2D inputTexture2;
uniform sampler2D inputTexture3;
uniform vec2 viewportSize;
uniform float zoom;
uniform int mode;
uniform float colorFactor1;
uniform int numTextures;

void main()
{
    vec4 color1 = texture(inputTexture1, texCoord);
    vec4 color2 = texture(inputTexture2, texCoord);
    vec3 finalColor = max(color1.rgb * colorFactor1, color2.rgb);
    if (numTextures >= 3) {
        vec4 color3 = texture(inputTexture3, texCoord);
        finalColor = max(finalColor.rgb, color3.rgb);
    }
    FragColor = vec4(finalColor, 1.0);
}
)";
}
