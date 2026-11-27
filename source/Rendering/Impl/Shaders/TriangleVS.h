#pragma once

#include <string_view>

namespace Shaders
{
    std::string_view const TriangleVS = R"(
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
uniform float renderScale;
uniform float solidBrightness;
uniform float cellBrightness;

const int ObjectType_Solid = 0;

void main()
{
    // Transform world position to normalized device coordinates
    vec2 relativePos = aPos.xy - rectUpperLeft;
    vec2 screenPos = relativePos * zoom;
    vec2 ndc = (screenPos / viewportSize) * 2.0 - 1.0;
    ndc.y = -ndc.y; // Flip Y coordinate
    
    gl_Position = vec4(ndc, aPos.z, 1.0);
    
    int objectType = (state >> 8) & 0xFF;
    bool isSolid = objectType == ObjectType_Solid;
    if (isSolid) {
        // Solids keep the brightness of their connections, so that their structure does not show
        vertexColor = aColor * solidBrightness;
    } else {
        // At zoom 3 there are not individual cells visible anymore, so no need to highlight the polygone more
        vertexColor = (zoom / renderScale < 3.0 ? aColor * 1.4 : aColor) * cellBrightness;
    }
    vertexSolid = isSolid ? 1.0 : 0.0;
}
)";
}
