#pragma once

#include <string_view>

namespace Shaders
{
    std::string_view const LineVS = R"(
#version 450
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;
layout (location = 2) in int state;

out vec3 vertexColor;
out float vertexSolid;

uniform vec2 worldSize;
uniform vec2 rectUpperLeft;
uniform float zoom;
uniform vec2 viewportSize;
uniform float solidBrightness;

const int ObjectType_Solid = 0;

// See Shaders/NonFluidObjectVS.h
float getMetaballBrightness(float brightness)
{
    return brightness * smoothstep(0.05, 0.55, brightness);
}

vec3 getSolidColor(vec3 color)
{
    float brightness = color.r + color.g + color.b;
    float target = brightness * solidBrightness;
    float low = 0.0;
    float high = brightness;
    if (getMetaballBrightness(high) <= target) {
        return color;
    }
    for (int i = 0; i < 12; ++i) {
        float middle = 0.5 * (low + high);
        if (getMetaballBrightness(middle) < target) {
            low = middle;
        } else {
            high = middle;
        }
    }
    return color * high / max(brightness, 1e-4);
}

void main()
{
    // Transform world position to screen coordinates
    // Note: Transform to normalized device coordinates is done in geometry shader.
    vec2 relativePos = aPos.xy - rectUpperLeft;
    vec2 screenPos = relativePos * zoom;
    
    gl_Position = vec4(screenPos, aPos.z, 1.0);
    
    // Pass color to geometry shader
    int objectType = (state >> 8) & 0xFF;
    bool isSolid = objectType == ObjectType_Solid;
    vertexColor = isSolid ? getSolidColor(aColor) : aColor;
    vertexSolid = isSolid ? 1.0 : 0.0;
}
)";
}
