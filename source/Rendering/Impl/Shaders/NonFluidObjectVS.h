#pragma once

#include <string_view>

namespace Shaders
{
    std::string_view const NonFluidObjectVS = R"(
#version 450
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;
layout (location = 2) in int state;
layout (location = 3) in float highlightIntensity;

out vec3 vColor;

uniform vec2 worldSize;
uniform vec2 rectUpperLeft;
uniform float zoom;
uniform float radius;
uniform vec2 viewportSize;
uniform float renderScale;
uniform float sizeScale;
uniform float solidBrightness;
uniform float solidDetailVisibility;

const int ObjectType_Solid = 0;
const int ObjectType_Fluid = 1;
const float DetailZoom = 5.0;

// The solids can be dimmed, see SimulationRenderer.cpp. Since the metaballs step would shrink dim solids, they are only dimmed down to this
// brightness.
const float MinDimmedSolidBrightness = 0.6;

vec3 getSolidColor(vec3 color)
{
    float brightness = color.r + color.g + color.b;
    float dimmedBrightness = max(brightness * solidBrightness, min(brightness, MinDimmedSolidBrightness));
    return color * dimmedBrightness / max(brightness, 1e-4);
}

void main()
{
    int objectType = (state >> 8) & 0xFF;
    if (objectType == ObjectType_Fluid) {
        gl_Position = vec4(-2.0, -2.0, -2.0, 1.0);
        gl_PointSize = 0.0;
        vColor = vec3(0.0);
        return;
    }

    // Transform world position to normalized device coordinates
    vec2 relativePos = aPos.xy - rectUpperLeft;
    vec2 screenPos = relativePos * zoom;
    vec2 ndc = (screenPos / viewportSize) * 2.0 - 1.0;
    ndc.y = -ndc.y; // Flip Y coordinate
    
    // Cells are rendered in front of lines (apply negative bias to bring forward)
    gl_Position = vec4(ndc, aPos.z, 1.0);

    float screenZoom = zoom / renderScale;

    bool isIsolatedOrTail = ((state >> 16) & 0x1) == 1;

    // Connected solids form smooth areas, their particles only appear close up
    float visibility = !isIsolatedOrTail && objectType == ObjectType_Solid ? solidDetailVisibility : 1.0;
    if ((!isIsolatedOrTail && screenZoom < DetailZoom) || visibility <= 0.0) {
        // Discard cells that have connections when zoomed out to avoid Moire patterns
        gl_Position = vec4(-2.0, -2.0, -2.0, 1.0);
    }
    float sizeFactor = isIsolatedOrTail ? max(1.0, min(2.0, screenZoom * 2)): 1.0;

    float visibleHighlightIntensity = screenZoom < DetailZoom ? 0.0 : highlightIntensity;
    vColor = mix(aColor, vec3(1.0), visibleHighlightIntensity * 0.2);
    if (objectType == ObjectType_Solid) {
        vColor = getSolidColor(vColor) * visibility;
    }
    gl_PointSize = radius * (0.4 + visibleHighlightIntensity * 0.2) * sizeFactor * sizeScale;
}
)";
}
