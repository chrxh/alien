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
    // based on cudaDrawRepetition from GeometryKernels.cu
    
    if (!borderlessRendering) {
        // If borderless rendering is disabled, just pass through the input
        FragColor = texture(inputTexture1, texCoord);
        return;
    }
    
    // Convert texture coordinates to screen position (in pixels)
    vec2 screenPos = texCoord * viewportSize;
    
    // Calculate universe image size (world size in pixels)
    vec2 universeImageSize = worldSize * zoom;
    
    // Check if we're outside the universe bounds
    if (screenPos.x >= universeImageSize.x || screenPos.y >= universeImageSize.y) {
        // We're outside the universe, need to copy from wrapped position
        
        // Convert screen position to world position
        vec2 worldPos = screenPos / zoom + rectUpperLeft;
        
        // Map world position back to image position with modulo wrapping
        vec2 refPos = (worldPos - rectUpperLeft) * zoom;
        refPos.x = modulo(refPos.x, universeImageSize.x);
        refPos.y = modulo(refPos.y, universeImageSize.y);
        
        // Convert back to texture coordinates
        vec2 refTexCoord = refPos / viewportSize;
        
        // Check if reference position is valid
        if (refPos.x >= 0.0 && refPos.x < viewportSize.x && 
            refPos.y >= 0.0 && refPos.y < viewportSize.y) {
            FragColor = texture(inputTexture1, refTexCoord);
        } else {
            // Invalid reference, keep original
            FragColor = texture(inputTexture1, texCoord);
        }
    } else {
        // We're inside the universe bounds, just pass through
        FragColor = texture(inputTexture1, texCoord);
    }
}
)";
}
