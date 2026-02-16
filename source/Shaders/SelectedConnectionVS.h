#pragma once

#include <string_view>

namespace Shaders
{
    std::string_view const SelectedConnectionVS = R"(
#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec3 aColor;
layout (location = 2) in float aConnectionWeightToObject1;
layout (location = 3) in float aConnectionWeightToObject2;

out vec3 vertexColor;
flat out float weightToObject1;
flat out float weightToObject2;

uniform vec2 worldSize;
uniform vec2 rectUpperLeft;
uniform float zoom;
uniform vec2 viewportSize;

void main()
{
    // Transform world position to screen coordinates
    // Note: Transform to normalized device coordinates (screenPos / viewportSize) * 2.0 - 1.0 is done in geometry shader.
    vec2 relativePos = aPos - rectUpperLeft;
    vec2 screenPos = relativePos * zoom;
	vec2 ndc = screenPos;
    ndc.y = ndc.y; // Flip Y coordinate
    
    gl_Position = vec4(ndc, 0.0, 1.0);
    
    // Pass color and connection weights to geometry shader
    vertexColor = aColor;
    weightToObject1 = aConnectionWeightToObject1;
    weightToObject2 = aConnectionWeightToObject2;
}
)";
}
