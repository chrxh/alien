#pragma once

#include <string_view>

namespace Shaders
{
    std::string_view const BlurHorizontalFS = R"(
#version 450
out vec4 FragColor;

in vec2 texCoord;

uniform sampler2D inputTexture1;
uniform vec2 viewportSize;
uniform float zoom;
uniform float strength;
uniform bool zoomDependent;
uniform float renderScale;

void main()
{
    vec2 texelSize = 1.0 / viewportSize;
    vec4 result = vec4(0.0);
    float totalWeight = 0.0;
    float blurRadius = zoomDependent ? zoom * strength : strength * renderScale;
    blurRadius = min(blurRadius, 20.0 * renderScale);
    
    if (blurRadius < 1.0) {
        FragColor = texture(inputTexture1, texCoord);
    } else {
        int radius = max(1, int(ceil(blurRadius)));
        float exponentFactor = -0.5 / (blurRadius * blurRadius);
        result = texture(inputTexture1, texCoord);
        totalWeight = 1.0;

        // Neighboring texels are fetched in pairs by a linear lookup between them that weights them like the Gaussian
        for (int distance = 1; distance <= radius; distance += 2) {
            float weight1 = exp(exponentFactor * float(distance * distance));
            float weight2 = distance < radius ? exp(exponentFactor * float((distance + 1) * (distance + 1))) : 0.0;
            float weight = weight1 + weight2;
            vec2 offset = vec2((float(distance) + weight2 / weight) * texelSize.x, 0.0);
            result += (texture(inputTexture1, texCoord + offset) + texture(inputTexture1, texCoord - offset)) * weight;
            totalWeight += 2.0 * weight;
        }

        // Normalize by total weight
        FragColor = result / totalWeight;
    }
}
)";
}
