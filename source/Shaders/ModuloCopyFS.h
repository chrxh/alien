#pragma once

#include <string_view>

namespace Shaders
{
    std::string_view const ModuloCopyFS = R"(
#version 330 core
out vec4 FragColor;

in vec2 texCoord;

uniform sampler2D inputTexture1;
uniform vec2 viewportSize;
uniform vec2 worldSize;
uniform vec2 rectUpperLeft;
uniform float zoom;
uniform bool borderlessRendering;

float modulo(float x, float y) {
    return x - y * floor(x / y);
}

void main()
{
    // This shader implements periodic copying of the input texture
    if (!borderlessRendering) {
        // If borderless rendering is disabled, just pass through the input
        FragColor = texture(inputTexture1, vec2(texCoord.x, 1.0 - texCoord.y));
        return;
    }
    
    // Convert texture coordinates to screen position (in pixels)
    vec2 screenPos = texCoord * viewportSize;
    
    // Calculate universe image size (world size in pixels)
    vec2 universeImageSize = worldSize * zoom;
    
    FragColor = vec4(1.0);
    
    // Convert screen position to world position
    vec2 worldPos = screenPos / zoom + rectUpperLeft;
    
    // Map world position back to image position with modulo wrapping
    vec2 refPos = (worldPos - rectUpperLeft) * zoom;
    refPos.x = modulo(refPos.x, universeImageSize.x);
    refPos.y = modulo(refPos.y, universeImageSize.y);
    refPos = clamp(refPos, vec2(0.0), universeImageSize - vec2(0.0));
    
    // Convert back to texture coordinates
    vec2 refTexCoord = refPos / viewportSize;
    
    FragColor = texture(inputTexture1, vec2(refTexCoord.x, 1.0 - refTexCoord.y));
}
)";
}
