#pragma once

#include <string_view>

namespace Shaders
{
    std::string_view const FresnelFS = R"(#version 330 core
out vec4 FragColor;

in vec2 texCoord;

uniform sampler2D inputTexture1;
uniform vec2 viewportSize;
uniform float zoom;

void main()
{
    // Sample the input texture (output from subsurface scattering)
    vec4 color = texture(inputTexture1, texCoord);
    
    // Calculate brightness as thickness for Fresnel effect
    float brightness = dot(color.rgb, vec3(0.333, 0.333, 0.333));
    
    // Fresnel effect: edges (thinner areas) reflect more light
    // Sample the gradient to detect edges
    vec2 texelSize = 1.0 / viewportSize;
    
    // Calculate gradient using Sobel-like operator
    float gradX = 0.0;
    float gradY = 0.0;
    
    // Sample surrounding pixels
    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            vec2 offset = vec2(x, y) * texelSize;
            float sample = dot(texture(inputTexture1, texCoord + offset).rgb, vec3(0.333, 0.333, 0.333));
            
            // Sobel kernel for X direction
            float kx = float(x);
            gradX += sample * kx;
            
            // Sobel kernel for Y direction
            float ky = float(y);
            gradY += sample * ky;
        }
    }
    
    // Calculate gradient magnitude (edge strength)
    float edgeStrength = length(vec2(gradX, gradY));
    
    // Apply Fresnel effect: enhance edges with a subtle highlight
    // Fresnel intensity is based on edge strength and inverse thickness
    float fresnelIntensity = edgeStrength * (1.0 - brightness * 0.5) * 0.5 * zoom / 40.0;
    
    // Add Fresnel highlight to the color
    vec3 fresnelColor = color.rgb + vec3(fresnelIntensity);
    
    FragColor = vec4(fresnelColor, 1.0f);
}
)";
}
