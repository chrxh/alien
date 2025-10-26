#pragma once

#include <string_view>

namespace Shaders
{
    std::string_view const CellTypeOverlayFS = R"(
#version 330 core
out vec4 FragColor;

in vec3 gColor;
in vec2 gQuadCoord;
in vec2 gTexCoord;
in vec2 gWorldPos;

uniform float zoom;
uniform vec2 worldSize;
uniform float radius;
uniform sampler2D overlayTexture;

void main()
{
    // Sample from the overlay texture atlas
    vec4 textColor = texture(overlayTexture, gTexCoord);
    
    // Only show if there's actually text content (alpha > 0)
    // if (textColor.a < 0.01) {
    //     discard;
    // } else if (textColor.r < 0.8) {
    //     FragColor = vec4(0.0, 0.0, 0.0, 0.5);
    // } else {
    //     FragColor = vec4(1.0, 1.0, 1.0, textColor.a * 0.5);
    // }
    FragColor = textColor;
})";
}
