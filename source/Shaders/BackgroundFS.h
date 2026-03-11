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
uniform bool borderlessRendering;

void main()
{
    // Convert texture coordinates to screen position (in pixels)
    vec2 screenPos = texCoord * viewportSize;
    
    // Convert screen position to world position
    vec2 relativePos = screenPos / zoom;
    vec2 worldPos = relativePos + rectUpperLeft;
    
    // Check if world position is within world boundaries
    vec3 finalColor;
    if (borderlessRendering ||
        (worldPos.x >= 0.0 && worldPos.x <= worldSize.x &&
         worldPos.y >= 0.0 && worldPos.y <= worldSize.y)) {

        // Inside world boundaries - render background color
        finalColor = background;
    } else {
        // Outside world boundaries - render black
        finalColor = vec3(0.0, 0.0, 0.0);
    }
    
    FragColor = vec4(finalColor, 1.0);
}
)";
}
