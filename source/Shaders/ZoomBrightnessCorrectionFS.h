#pragma once

#include <string_view>

namespace Shaders
{
    std::string_view const ZoomBrightnessCorrectionFS = R"(
#version 330 core
out vec4 FragColor;

in vec2 texCoord;

uniform sampler2D inputTexture1;
uniform vec2 viewportSize;
uniform float zoom;
uniform float strength;

void main()
{
    vec2 texelSize = 1.0 / viewportSize;
    
    FragColor = texture(inputTexture1, texCoord) * min(1.0, zoom * strength);
}
)";
}
