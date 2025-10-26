#pragma once

#include <string_view>

namespace Shaders
{
    std::string_view const BackgroundFS = R"(
#version 330 core
out vec4 FragColor;

in vec2 texCoord;

uniform vec2 viewportSize;
uniform float zoom;
uniform vec3 background;
uniform vec2 worldSize;
uniform vec2 rectUpperLeft;
uniform vec2 rectLowerRight;
uniform bool gridLines;

float modulo(float x, float y) {
    return x - y * floor(x / y);
}

void main()
{
    // Convert texture coordinates to screen position (in pixels)
    vec2 screenPos = texCoord * viewportSize;
    
    // Convert screen position to world position
    vec2 relativePos = screenPos / zoom;
    vec2 worldPos = relativePos + rectUpperLeft;
    
    // Check if world position is within world boundaries
    vec3 finalColor;
    if (worldPos.x >= 0.0 && worldPos.x <= worldSize.x &&
        worldPos.y >= 0.0 && worldPos.y <= worldSize.y) {
        // Inside world boundaries - render background color
        finalColor = background;
    } else {
        // Outside world boundaries - render black
        finalColor = vec3(0.0, 0.0, 0.0);
    }
    
    // Add grid lines if enabled
    if (gridLines) {
        // Calculate grid parameters (matching CUDA code logic)
        float viewWidth = max(1.0, rectLowerRight.x - rectUpperLeft.x);
        float PixelInWorldSize = viewWidth / worldSize.x;
        float gridDistance = pow(10.0, floor(log(viewWidth) / log(10.0))) / 10.0;
        float maxGridDistance = viewWidth / 10.0;
        float gridRemainder = (maxGridDistance - gridDistance) / maxGridDistance;
        
        // Coarse grid (primary grid lines)
        {
            float distanceX = modulo(worldPos.x + gridDistance / 2.0, gridDistance) - gridDistance / 2.0;
            float distanceY = modulo(worldPos.y + gridDistance / 2.0, gridDistance) - gridDistance / 2.0;
            
            if (abs(distanceX) <= PixelInWorldSize * 8.0) {
                float viewDistance = max(0.0, 0.1 - abs(distanceX) * zoom / 10.0) * gridRemainder * 0.7;
                finalColor += vec3(viewDistance, viewDistance, viewDistance);
            }
            if (abs(distanceY) <= PixelInWorldSize * 8.0) {
                float viewDistance = max(0.0, 0.1 - abs(distanceY) * zoom / 10.0) * gridRemainder * 0.7;
                finalColor += vec3(viewDistance, viewDistance, viewDistance);
            }
        }
        
        // Fine grid (secondary grid lines)
        {
            float distanceX = modulo(worldPos.x + gridDistance / 20.0, gridDistance / 10.0) - gridDistance / 20.0;
            float distanceY = modulo(worldPos.y + gridDistance / 20.0, gridDistance / 10.0) - gridDistance / 20.0;
            
            if (abs(distanceX) <= PixelInWorldSize * 8.0) {
                float viewDistance = max(0.0, 0.1 - abs(distanceX) * zoom / 10.0) * (1.0 - gridRemainder) * 0.7;
                finalColor += vec3(viewDistance, viewDistance, viewDistance);
            }
            if (abs(distanceY) <= PixelInWorldSize * 8.0) {
                float viewDistance = max(0.0, 0.1 - abs(distanceY) * zoom / 10.0) * (1.0 - gridRemainder) * 0.7;
                finalColor += vec3(viewDistance, viewDistance, viewDistance);
            }
        }
    }
    
    FragColor = vec4(finalColor, 1.0);
}
)";
}
