#pragma once

#include <string_view>

namespace Shaders
{
    std::string_view const BlurHorizontalFS = R"(
#version 330 core
out vec4 FragColor;

in vec2 texCoord;

uniform sampler2D inputTexture1;
uniform vec2 viewportSize;
uniform float zoom;
uniform float strength;
uniform bool zoomDependent;

void main()
{
    vec2 texelSize = 1.0 / viewportSize;
    vec4 result = vec4(0.0);
    float totalWeight = 0.0;
    float blurRadius = zoomDependent ? zoom * strength : strength;
    blurRadius = min(blurRadius, 20.0);
    
    if (blurRadius < 1.0) {
        FragColor = texture(inputTexture1, texCoord);
    } else {
        int radius = max(1, int(ceil(blurRadius)));
        
        // Apply horizontal Gaussian blur
        for (int x = -radius; x <= radius; x++) {
            float distance = float(x);
            // Gaussian weight calculation
            float weight = exp(-0.5 * (distance * distance) / (blurRadius * blurRadius));
            vec2 offset = vec2(distance * texelSize.x, 0.0);
            result += texture(inputTexture1, texCoord + offset) * weight;
            totalWeight += weight;
        }
        
        // Normalize by total weight
        FragColor = result / totalWeight;
    }
}
)";
}
