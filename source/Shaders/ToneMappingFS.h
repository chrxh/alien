#pragma once

#include <string_view>

namespace Shaders
{
    std::string_view const ToneMappingFS = R"(
#version 330 core
out vec4 FragColor;

in vec2 texCoord;

uniform sampler2D inputTexture1;
uniform vec2 viewportSize;
uniform float zoom;

// Reinhard tone mapping
vec3 toneMapReinhard(vec3 color) {
    return color / (color + vec3(1.0));
}

// ACES filmic tone mapping (approximation)
vec3 toneMapACES(vec3 color) {
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    return clamp((color * (a * color + b)) / (color * (c * color + d) + e), 0.0, 1.0);
}

// Gamma correction
vec3 gammaCorrect(vec3 color, float gamma) {
    return pow(color, vec3(1.0 / gamma));
}

void main()
{
    vec4 color = texture(inputTexture1, texCoord);
    
    // Apply tone mapping (using ACES for better results with HDR content)
    vec3 toneMappedColor = toneMapACES(color.rgb);
    
    // Apply gamma correction (standard gamma 2.2)
    vec3 gammaCorrectedColor = gammaCorrect(toneMappedColor, 2.2);
    
    FragColor = vec4(gammaCorrectedColor, color.a);
}
)";
}
